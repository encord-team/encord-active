import React from "react";

interface ErrorBoundaryProps {
  children: React.ReactNode;
}
interface ErrorBoundaryState {
  error: Error | undefined;
}

/**
 * Shows errors thrown from child components.
 */
export class ErrorBoundary extends React.PureComponent<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { error: undefined };
  }

  static getDerivedStateFromError(error: Error) {
    // Update state so the next render will show the fallback UI.
    return { error };
  }

  render() {
    if (this.state.error != null) {
      return (
        <div>
          <h1>Component Error</h1>
          <span>{this.state.error.message}</span>
        </div>
      );
    }

    return this.props.children;
  }
}
