//! Timing of the planner, and its association with the later execution.
//!
//! # Design
//!
//! `planner_hook` measures every call of the planner and remembers the result
//! in a small backend-local table. When the statement finishes
//! (`ExecutorEnd`), the span builder takes the entry belonging to the executed
//! plan out of the table (once) and adds a `planning` span.
//!
//! The planner hook keeps **no state while the planner runs**: the start time
//! lives in local variables of the hook, and the table is only touched after
//! the planner returned successfully. An ERROR raised by the planner (which
//! unwinds through the hook) therefore cannot leave anything half-updated.
//! Nested planner calls (for example a function evaluated while constant
//! folding that plans its own statements) each measure on their own; the outer
//! duration includes the inner ones.
//!
//! ## Matching a plan with its planning time
//!
//! The obvious key, the address of the returned `PlannedStmt`, does not work:
//! the plan cache copies the plan list (`copyObject`) right after planning, so
//! the statement that is executed for prepared statements and for SPI/PL/pgSQL
//! saved plans is a different object. Instead an entry is keyed by a
//! [`PlanKey`], a fingerprint of values that the copy preserves exactly (query
//! id, statement position and length, and the cost estimates and row count of
//! the top plan node). Matching is by content, so an entry is **consumed once**:
//! a cached plan that is executed again finds nothing and gets no planning
//! span, which is correct, because it was not planned again.
//!
//! A fingerprint could in theory match a different plan with identical values.
//! The table is kept small and clean to make that irrelevant in practice:
//!
//! * entries are removed when the statement ends, when `EXPLAIN` (without
//!   `ANALYZE`) planned a statement that is never executed, at the end of the
//!   (sub)transaction that created them, and the oldest entry is evicted when
//!   there are more than [`MAX_PENDING`];
//! * a plan created in an earlier transaction and executed later simply has no
//!   planning span.

use std::{
    cell::RefCell,
    collections::VecDeque,
    ffi::c_void,
    ptr,
    time::{Instant, SystemTime},
};

use pgrx::pg_sys;

use crate::postgres::tracing_enabled_here;

/// Most planner results remembered at once. Statements that are planned but
/// never executed (`EXPLAIN`, cursors never fetched, ...) would otherwise pile
/// up within a long transaction.
pub const MAX_PENDING: usize = 64;

/// Fingerprint of a plan that survives the plan cache's `copyObject`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanKey {
    query_id: i64,
    stmt_location: i32,
    stmt_len: i32,
    startup_cost_bits: u64,
    total_cost_bits: u64,
    plan_rows_bits: u64,
    plan_width: i32,
}

impl PlanKey {
    /// # Safety
    ///
    /// `stmt` must be null or point to a valid `PlannedStmt`.
    pub unsafe fn of(stmt: *const pg_sys::PlannedStmt) -> Option<Self> {
        // SAFETY: null or valid per the function contract.
        let stmt = unsafe { stmt.as_ref()? };
        // SAFETY: a planned statement's plan tree is null or valid. Utility
        // statements have none and are never timed.
        let plan = unsafe { stmt.planTree.as_ref()? };
        Some(Self {
            query_id: stmt.queryId,
            stmt_location: stmt.stmt_location,
            stmt_len: stmt.stmt_len,
            startup_cost_bits: plan.startup_cost.to_bits(),
            total_cost_bits: plan.total_cost.to_bits(),
            plan_rows_bits: plan.plan_rows.to_bits(),
            plan_width: plan.plan_width,
        })
    }
}

/// When planning started and how long it took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanningTiming {
    /// Wall-clock start (for the span).
    pub start: SystemTime,
    /// Duration, measured with a monotonic clock.
    pub duration_ns: i64,
}

#[derive(Debug)]
struct Entry {
    key: PlanKey,
    /// The sub-transaction the statement was planned in.
    subxact: u32,
    timing: PlanningTiming,
}

/// Bounded table of planner results waiting for their execution.
#[derive(Debug, Default)]
pub struct PendingPlans {
    entries: VecDeque<Entry>,
}

impl PendingPlans {
    pub const fn new() -> Self {
        Self {
            entries: VecDeque::new(),
        }
    }

    /// Remembers `timing` for `key`, replacing an older entry with the same key
    /// (the newer planning is the one that will be executed) and evicting the
    /// oldest entry when the table is full.
    pub fn insert(&mut self, key: PlanKey, subxact: u32, timing: PlanningTiming) {
        if let Some(position) = self.entries.iter().position(|entry| entry.key == key) {
            self.entries.remove(position);
        }
        if self.entries.len() >= MAX_PENDING {
            self.entries.pop_front();
        }
        self.entries.push_back(Entry {
            key,
            subxact,
            timing,
        });
    }

    /// Removes and returns the entry for `key`.
    pub fn take(&mut self, key: &PlanKey) -> Option<PlanningTiming> {
        let position = self.entries.iter().rposition(|entry| entry.key == *key)?;
        self.entries.remove(position).map(|entry| entry.timing)
    }

    /// Forgets everything (end of a transaction).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Forgets the entries created in the aborted sub-transaction
    /// `aborted_subxact` or in any sub-transaction started inside it. Ids grow
    /// with start order, so those all have an id of at least `aborted_subxact`.
    pub fn forget_subxact(&mut self, aborted_subxact: u32) {
        self.entries.retain(|entry| entry.subxact < aborted_subxact);
    }

    #[cfg(any(test, feature = "pg_test"))]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

thread_local! {
    static PENDING: RefCell<PendingPlans> = const { RefCell::new(PendingPlans::new()) };
}

/// Runs the planner through `plan` and records how long it took.
///
/// Does nothing but call `plan` when tracing is disabled (or in a parallel
/// worker). The monotonic [`Instant`] is used for the duration so clock
/// adjustments cannot produce negative or inflated values; the wall clock is
/// only read once, for the span's start timestamp.
pub fn timed(plan: impl FnOnce() -> *mut pg_sys::PlannedStmt) -> *mut pg_sys::PlannedStmt {
    if !tracing_enabled_here() {
        return plan();
    }
    let start = SystemTime::now();
    let started = Instant::now();
    let stmt = plan();
    let duration = started.elapsed();
    // SAFETY: `stmt` was just returned by the planner.
    if let Some(key) = unsafe { PlanKey::of(stmt) } {
        let timing = PlanningTiming {
            start,
            duration_ns: i64::try_from(duration.as_nanos()).unwrap_or(i64::MAX),
        };
        // SAFETY: plain read of a backend-local variable.
        let subxact = unsafe { pg_sys::GetCurrentSubTransactionId() };
        PENDING.with_borrow_mut(|pending| pending.insert(key, subxact, timing));
    }
    stmt
}

/// Takes the planning time of the plan that `query_desc` executes, if it was
/// planned in this transaction and not taken before.
///
/// # Safety
///
/// `query_desc` must be null or point to a valid `QueryDesc`.
pub unsafe fn take_for(query_desc: *const pg_sys::QueryDesc) -> Option<PlanningTiming> {
    // SAFETY: null or valid per the function contract.
    let stmt = unsafe { query_desc.as_ref()? }.plannedstmt;
    // SAFETY: the executed statement is valid.
    let key = unsafe { PlanKey::of(stmt)? };
    PENDING.with_borrow_mut(|pending| pending.take(&key))
}

/// Number of remembered planner results; for tests.
#[cfg(any(test, feature = "pg_test"))]
pub fn pending_len() -> usize {
    PENDING.with_borrow(PendingPlans::len)
}

/// Whether a transaction event is the end of the top-level transaction.
fn ends_transaction(event: pg_sys::XactEvent::Type) -> bool {
    matches!(
        event,
        pg_sys::XactEvent::XACT_EVENT_COMMIT
            | pg_sys::XactEvent::XACT_EVENT_ABORT
            | pg_sys::XactEvent::XACT_EVENT_PREPARE
    )
}

unsafe extern "C-unwind" fn on_xact_event(event: pg_sys::XactEvent::Type, _arg: *mut c_void) {
    if ends_transaction(event) {
        // Cannot fail: the table is only borrowed for short, non-reentrant
        // operations.
        PENDING.with_borrow_mut(PendingPlans::clear);
    }
}

unsafe extern "C-unwind" fn on_subxact_event(
    event: pg_sys::SubXactEvent::Type,
    my_subid: pg_sys::SubTransactionId,
    _parent_subid: pg_sys::SubTransactionId,
    _arg: *mut c_void,
) {
    if event == pg_sys::SubXactEvent::SUBXACT_EVENT_ABORT_SUB {
        PENDING.with_borrow_mut(|pending| pending.forget_subxact(my_subid));
    }
}

/// Registers the transaction callbacks that keep the table clean. Call once from
/// `_PG_init`.
pub fn register_callbacks() {
    // SAFETY: called once in the postmaster; the callbacks are plain functions
    // and need no argument.
    unsafe {
        pg_sys::RegisterXactCallback(Some(on_xact_event), ptr::null_mut());
        pg_sys::RegisterSubXactCallback(Some(on_subxact_event), ptr::null_mut());
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn key(n: i64) -> PlanKey {
        PlanKey {
            query_id: n,
            stmt_location: 0,
            stmt_len: 10,
            startup_cost_bits: 0,
            total_cost_bits: 1,
            plan_rows_bits: 2,
            plan_width: 4,
        }
    }

    fn timing(ns: i64) -> PlanningTiming {
        PlanningTiming {
            start: SystemTime::UNIX_EPOCH + Duration::from_secs(1),
            duration_ns: ns,
        }
    }

    #[test]
    fn an_entry_is_consumed_once() {
        let mut pending = PendingPlans::new();
        pending.insert(key(1), 1, timing(10));
        assert_eq!(pending.take(&key(1)), Some(timing(10)));
        assert_eq!(pending.take(&key(1)), None);
        assert_eq!(pending.len(), 0);
    }

    #[test]
    fn entries_of_other_plans_are_untouched() {
        let mut pending = PendingPlans::new();
        pending.insert(key(1), 1, timing(10));
        pending.insert(key(2), 1, timing(20));
        assert_eq!(pending.take(&key(3)), None);
        assert_eq!(pending.take(&key(2)), Some(timing(20)));
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn planning_the_same_plan_again_replaces_the_entry() {
        let mut pending = PendingPlans::new();
        pending.insert(key(1), 1, timing(10));
        pending.insert(key(1), 1, timing(99));
        assert_eq!(pending.len(), 1);
        assert_eq!(pending.take(&key(1)), Some(timing(99)));
    }

    #[test]
    fn the_oldest_entry_is_evicted_when_full() {
        let mut pending = PendingPlans::new();
        for n in 0..MAX_PENDING as i64 {
            pending.insert(key(n), 1, timing(n));
        }
        assert_eq!(pending.len(), MAX_PENDING);
        pending.insert(key(1000), 1, timing(1000));
        assert_eq!(pending.len(), MAX_PENDING);
        assert_eq!(pending.take(&key(0)), None, "oldest was evicted");
        assert_eq!(pending.take(&key(1)), Some(timing(1)));
        assert_eq!(pending.take(&key(1000)), Some(timing(1000)));
    }

    #[test]
    fn replacing_an_entry_does_not_evict_others() {
        let mut pending = PendingPlans::new();
        for n in 0..MAX_PENDING as i64 {
            pending.insert(key(n), 1, timing(n));
        }
        pending.insert(key(5), 1, timing(500));
        assert_eq!(pending.len(), MAX_PENDING);
        assert_eq!(pending.take(&key(0)), Some(timing(0)));
    }

    #[test]
    fn clear_forgets_everything() {
        let mut pending = PendingPlans::new();
        pending.insert(key(1), 1, timing(1));
        pending.insert(key(2), 2, timing(2));
        pending.clear();
        assert!(pending.is_empty());
        assert_eq!(pending.take(&key(1)), None);
    }

    #[test]
    fn an_aborted_subtransaction_forgets_its_own_and_nested_entries_only() {
        let mut pending = PendingPlans::new();
        pending.insert(key(1), 1, timing(1)); // top level
        pending.insert(key(2), 2, timing(2)); // the aborted subtransaction
        pending.insert(key(3), 3, timing(3)); // started inside it
        pending.forget_subxact(2);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending.take(&key(1)), Some(timing(1)));
    }

    #[test]
    fn only_the_end_of_a_transaction_clears_the_table() {
        assert!(ends_transaction(pg_sys::XactEvent::XACT_EVENT_COMMIT));
        assert!(ends_transaction(pg_sys::XactEvent::XACT_EVENT_ABORT));
        assert!(ends_transaction(pg_sys::XactEvent::XACT_EVENT_PREPARE));
        assert!(!ends_transaction(pg_sys::XactEvent::XACT_EVENT_PRE_COMMIT));
        assert!(!ends_transaction(
            pg_sys::XactEvent::XACT_EVENT_PARALLEL_COMMIT
        ));
    }
}
