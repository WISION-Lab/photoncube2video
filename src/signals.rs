use anyhow::Result;
use pyo3::{AsPyPointer, PyAny, Python};

/// Effectively a context manager telling python to temporarily ignore
/// and CTRL-C/SIGINTs and let rust handle them.  
///
/// Usage:
///     let _defer = DeferedSignal::new(py)?;
///
/// Default handler is restored when `_defer` is dropped, which will usually
/// happen when it fals out of scope. Due to this you _must_ assign it to a
/// variable, otherwise `DeferedSignal` will be created and dropped immediately.
pub struct DeferedSignal<'py> {
    set_sig: Option<&'py PyAny>,
    signal: Option<&'py PyAny>,
    default_handler: Option<&'py PyAny>,
    signal_name: String,
    is_main_thread: bool,
}

impl<'py> DeferedSignal<'py> {
    /// Set SIGINT to defer default action sucj that rust can intercep it.
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
            let signal = signal_mod.getattr(signal_name)?;
            let default_handler = signal_mod.getattr("getsignal")?.call1((signal,))?;
            let set_sig = signal_mod.getattr("signal")?;

            set_sig
                .call1((signal, signal_mod.getattr("SIG_DFL")?))
                .map(|_| ())?;
            Ok(DeferedSignal {
                set_sig: Some(set_sig),
                signal: Some(signal),
                default_handler: Some(default_handler),
                signal_name: signal_name.to_string(),
                is_main_thread,
            })
        } else {
            Ok(DeferedSignal {
                set_sig: None,
                signal: None,
                default_handler: None,
                signal_name: signal_name.to_string(),
                is_main_thread,
            })
        }
    }
}

impl<'py> Drop for DeferedSignal<'py> {
    /// Restore default SIGINT handler when object goes out of scope.
    fn drop(&mut self) {
        if self.is_main_thread {
            self.set_sig
                .unwrap()
                .call1((self.signal.unwrap(), self.default_handler.unwrap()))
                .map(|_| ())
                .expect("Unable to restore default SIGINT handler.");
        }
    }
}

impl<'py> std::fmt::Display for DeferedSignal<'py> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "DeferedSignal({}, is_main_thread: {})",
            self.signal_name, self.is_main_thread
        )
    }
}
