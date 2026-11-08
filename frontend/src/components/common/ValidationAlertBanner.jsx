import React from 'react';
import { useEngine } from '../../context/EngineContext';

/**
 * ValidationAlertBanner Component
 * Phase 8 - Step 8.10: High-visibility defensive alert banner for mission control.
 * Displays categorised error messages for:
 * - INVALID_ENGINE_ID (bounds, out-of-range engines, cycle exceedance)
 * - MISSING_INPUT (HTTP 422, unpopulated forms, malformed payload tensors)
 * - MODEL_UNAVAILABLE (HTTP 503, missing/corrupted checkpoints/scalers)
 */
export const ValidationAlertBanner = ({
  errorOverride = null,
  onDismiss = null,
}) => {
  const { validationError, clearValidationError } = useEngine();

  const currentError = errorOverride || validationError;
  if (!currentError) return null;

  const {
    category = 'INVALID_ENGINE_ID',
    severity = 'ERROR',
    title = 'System Validation Alert',
    message = 'An operational parameter validation error occurred.',
    detail = null,
    suggestion = null,
  } = currentError;

  const handleDismiss = () => {
    if (onDismiss) {
      onDismiss();
    } else if (clearValidationError) {
      clearValidationError();
    }
  };

  // Color theme per severity
  const severityStyles = {
    ERROR: {
      border: 'border-rose-700/70',
      bg: 'bg-rose-950/60',
      text: 'text-rose-200',
      badgeBg: 'bg-rose-900/80 text-rose-200 border-rose-600',
      icon: '⛔',
    },
    WARNING: {
      border: 'border-amber-700/70',
      bg: 'bg-amber-950/60',
      text: 'text-amber-200',
      badgeBg: 'bg-amber-900/80 text-amber-200 border-amber-600',
      icon: '⚠',
    },
    INFO: {
      border: 'border-cyan-700/70',
      bg: 'bg-cyan-950/60',
      text: 'text-cyan-200',
      badgeBg: 'bg-cyan-900/80 text-cyan-200 border-cyan-600',
      icon: 'ℹ',
    },
  }[severity] || {
    border: 'border-rose-700/70',
    bg: 'bg-rose-950/60',
    text: 'text-rose-200',
    badgeBg: 'bg-rose-900/80 text-rose-200 border-rose-600',
    icon: '⛔',
  };

  // Human-readable category labels
  const categoryLabels = {
    INVALID_ENGINE_ID: 'INVALID ENGINE IDENTIFIER',
    MISSING_INPUT: 'MISSING OR MALFORMED INPUT',
    MODEL_UNAVAILABLE: 'CHAMPION MODEL UNAVAILABLE',
  };

  return (
    <div
      role="alert"
      aria-live="assertive"
      className={`rounded-xl border ${severityStyles.border} ${severityStyles.bg} p-4 shadow-xl space-y-2.5 transition-all duration-200 animate-fadeIn`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5 flex-wrap">
          <span className="text-base select-none">{severityStyles.icon}</span>

          {/* Category Tag */}
          <span
            className={`px-2.5 py-0.5 rounded font-mono font-bold text-[11px] border tracking-wider uppercase ${severityStyles.badgeBg}`}
          >
            {categoryLabels[category] || category}
          </span>

          {/* Severity Tag */}
          <span className="text-[10px] font-mono px-2 py-0.5 rounded border border-slate-700 bg-slate-900 text-slate-400 uppercase">
            SEVERITY: {severity}
          </span>

          <h3 className={`text-sm font-bold tracking-wide ${severityStyles.text}`}>
            {title}
          </h3>
        </div>

        {/* Dismiss Button */}
        <button
          type="button"
          onClick={handleDismiss}
          aria-label="Dismiss Alert"
          className="text-slate-400 hover:text-white transition-colors p-1 rounded hover:bg-white/10 text-xs font-mono select-none"
        >
          ✕ Dismiss
        </button>
      </div>

      {/* Main Message Body */}
      <div className="text-xs text-slate-300 leading-relaxed font-sans pl-6">
        <p>{message}</p>

        {/* Detailed Context / Bounds */}
        {detail && (
          <div className="mt-2 p-2 rounded bg-black/40 border border-white/10 font-mono text-[11px] text-slate-300 max-h-32 overflow-y-auto">
            {typeof detail === 'string' ? (
              <span>{detail}</span>
            ) : Array.isArray(detail) ? (
              <ul className="list-disc list-inside space-y-0.5">
                {detail.map((d, i) => (
                  <li key={i}>
                    {typeof d === 'object' ? JSON.stringify(d) : String(d)}
                  </li>
                ))}
              </ul>
            ) : (
              <pre>{JSON.stringify(detail, null, 2)}</pre>
            )}
          </div>
        )}

        {/* Actionable Remediation Guidance */}
        {suggestion && (
          <div className="mt-2 text-[11px] font-mono text-cyan-300 flex items-start gap-1.5 bg-cyan-950/30 p-2 rounded border border-cyan-800/40">
            <span className="font-bold select-none">Remediation:</span>
            <span>{suggestion}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ValidationAlertBanner;
