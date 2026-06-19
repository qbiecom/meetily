'use client';

import { useEffect, useMemo, useState } from 'react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const currentUrl = typeof window !== 'undefined' ? window.location.href : 'Unavailable';
  const stack = error.stack || 'No stack trace available';
  const details = useMemo(() => {
    return [
      `Message: ${error.message || 'Unknown error'}`,
      `Digest: ${error.digest || 'None'}`,
      `URL: ${currentUrl}`,
      '',
      stack,
    ].join('\n');
  }, [currentUrl, error.digest, error.message, stack]);

  useEffect(() => {
    console.error('[AppErrorBoundary] Route error', error, {
      digest: error.digest,
      href: currentUrl,
    });
  }, [currentUrl, error]);

  const handleCopyDetails = async () => {
    try {
      await navigator.clipboard.writeText(details);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch (copyError) {
      console.error('Failed to copy error details', copyError);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-50 p-6 text-gray-900">
      <div className="w-full max-w-2xl rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <div className="mb-5">
          <p className="mb-2 text-sm font-medium uppercase tracking-wide text-red-600">Something went wrong</p>
          <h1 className="text-2xl font-semibold text-gray-950">This page could not load</h1>
          <p className="mt-2 text-sm text-gray-600">
            You can retry the page, go back to the home screen, or copy the error details for troubleshooting.
          </p>
        </div>

        <div className="mb-4 rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm">
          <div className="mb-2">
            <span className="font-semibold text-gray-700">Message: </span>
            <span className="text-gray-900">{error.message || 'Unknown error'}</span>
          </div>
          <div className="mb-2">
            <span className="font-semibold text-gray-700">Digest: </span>
            <span className="text-gray-900">{error.digest || 'None'}</span>
          </div>
          <div className="break-all">
            <span className="font-semibold text-gray-700">URL: </span>
            <span className="text-gray-900">{currentUrl}</span>
          </div>
        </div>

        <details className="mb-5 rounded-lg border border-gray-200 bg-gray-950 text-gray-100">
          <summary className="cursor-pointer px-4 py-3 text-sm font-medium">Show technical details</summary>
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap border-t border-gray-800 p-4 text-xs leading-relaxed">
            {stack}
          </pre>
        </details>

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={reset}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700"
          >
            Retry
          </button>
          <button
            type="button"
            onClick={() => window.location.assign('/')}
            className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-800 transition-colors hover:bg-gray-50"
          >
            Go home
          </button>
          <button
            type="button"
            onClick={handleCopyDetails}
            className="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-800 transition-colors hover:bg-gray-50"
          >
            {copied ? 'Copied' : 'Copy details'}
          </button>
        </div>
      </div>
    </main>
  );
}
