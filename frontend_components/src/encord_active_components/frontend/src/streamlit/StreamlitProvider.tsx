import React, { useContext, useEffect } from "react";
import { RenderData, Streamlit } from "streamlit-component-lib";
import { ErrorBoundary } from "./ErrorBoundry";
import { useNullableRenderData } from "./UseNullableRenderData";

const renderDataContext = React.createContext<RenderData | undefined>(
  undefined
);

// A narrower type for streamlit RederData args
type StreamlitDefaultRenderArgs = {
  default: null | unknown;
  key: null | unknown;
};

/**
 * Returns `RenderData` received from Streamlit.
 * Accepts a generic args type definition (NOTE: A `Partial` would be applied
 * on `args` since we can't guarantee they will actually be sent)
 */
export const useRenderData = <T extends Record<string, any>>(): Omit<
  RenderData,
  "args"
> & { args: Partial<T> & StreamlitDefaultRenderArgs } => {
  const contextValue = useContext(renderDataContext);
  if (contextValue == null) {
    throw new Error(
      "useRenderData() must be used inside <StreamlitProvider />"
    );
  }

  return contextValue;
};

/**
 * Wrapper for React-hooks-based Streamlit components.
 *
 * Bootstraps the communication interface between Streamlit and the component.
 */
export const StreamlitProvider = (props: { children: React.ReactNode }) => {
  const renderData = useNullableRenderData();

  useEffect(() => {
    Streamlit.setFrameHeight();
  });

  // Don't render until we've gotten our first data from Streamlit.
  if (renderData == null) {
    return null;
  }

  return (
    <ErrorBoundary>
      <renderDataContext.Provider value={renderData}>
        {props.children}
      </renderDataContext.Provider>
    </ErrorBoundary>
  );
};
