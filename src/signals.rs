use anyhow::Result;
use pyo3::{PyAny, Python};

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
    set_sig: &'py PyAny,
    signal: &'py PyAny,
    default_handler: &'py PyAny,
    signal_name: String,
}

impl<'py> DeferedSignal<'py> {
    /// Set SIGINT to defer default action sucj that rust can intercep it.
    /// Perform all error handling preemptively here and store refs in struct.
    pub fn new(py: Python<'py>, signal_name: &str) -> Result<Self> {
        let signal_mod = py.import("signal")?;
        let signal = signal_mod.getattr(signal_name)?;
        let default_handler = signal_mod.getattr("getsignal")?.call1((signal,))?;
        let set_sig = signal_mod.getattr("signal")?;

        set_sig
            .call1((signal, signal_mod.getattr("SIG_DFL")?))
            .map(|_| ())?;
        Ok(DeferedSignal {
            set_sig,
            signal,
            default_handler,
            signal_name: signal_name.to_string(),
        })
    }
}

impl<'py> Drop for DeferedSignal<'py> {
    /// Restore default SIGINT handler when object goes out of scope.
    fn drop(&mut self) {
        self.set_sig
            .call1((self.signal, self.default_handler))
            .map(|_| ())
            .expect("Unable to restore default SIGINT handler.");
    }
}

impl<'py> std::fmt::Display for DeferedSignal<'py> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "DeferedSignal({})", self.signal_name)
    }
}
