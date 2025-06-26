use anyhow::Result;
use pyo3::{types::PyAnyMethods, Bound, PyAny, Python};

/// Effectively a context manager telling python to temporarily ignore
/// and CTRL-C/SIGINTs and let rust handle them.  
///
/// Usage:
///     let _defer = DeferredSignal::new(py, "SIGINT")?;
///
/// Default handler is restored when `_defer` is dropped, which will usually
/// happen when it falls out of scope. Due to this you _must_ assign it to a
/// variable, otherwise `DeferredSignal` will be created and dropped immediately.
pub struct DeferredSignal<'py> {
    set_sig: Option<Bound<'py, PyAny>>,
    signal: Option<Bound<'py, PyAny>>,
    default_handler: Option<Bound<'py, PyAny>>,
    signal_name: String,
    is_main_thread: bool,
}

impl<'py> DeferredSignal<'py> {
    /// Set SIGINT to defer default action such that rust can intercept it.
    /// Perform all error handling preemptively here and store refs in struct.
    pub fn new(py: Python<'py>, signal_name: &str) -> Result<Self> {
        // Note: We cannot defer signals if not in Python's main thread, so if we
        //       detect we aren't we effectively NO-OP.
        let threading_mod = py.import("threading")?;
        let current_thread = threading_mod.call_method("current_thread", (), None)?;
        let main_thread = threading_mod.call_method("main_thread", (), None)?;
        let is_main_thread = current_thread.as_ptr() == main_thread.as_ptr(); // Equiv to `is` but backwards compat.

        if is_main_thread {
            let signal_mod = py.import("signal")?;
            let set_sig = signal_mod.getattr("signal")?;
            let signal = signal_mod.getattr(signal_name)?;
            let signal_id: u32 = signal.getattr("value")?.extract()?;
            let default_handler = set_sig.call1((signal_id, signal_mod.getattr("SIG_DFL")?))?;

            Ok(DeferredSignal {
                set_sig: Some(set_sig),
                signal: Some(signal),
                default_handler: Some(default_handler),
                signal_name: signal_name.to_string(),
                is_main_thread,
            })
        } else {
            Ok(DeferredSignal {
                set_sig: None,
                signal: None,
                default_handler: None,
                signal_name: signal_name.to_string(),
                is_main_thread,
            })
        }
    }
}

impl Drop for DeferredSignal<'_> {
    /// Restore default SIGINT handler when object goes out of scope.
    fn drop(&mut self) {
        if self.is_main_thread {
            self.set_sig
                .as_ref()
                .unwrap()
                .call1((
                    self.signal.as_ref().unwrap(),
                    self.default_handler.as_ref().unwrap(),
                ))
                .map(|_| ())
                .expect("Unable to restore default SIGINT handler.");
        }
    }
}

impl std::fmt::Display for DeferredSignal<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "DeferredSignal({}, is_main_thread: {})",
            self.signal_name, self.is_main_thread
        )
    }
}
