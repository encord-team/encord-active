import { useEffect, useRef, useState } from "react";

function useRateLimit<T>(value: T, delay: number): T {
  const latestValue = useRef<T>(value);
  const [rateLimitValue, setRateLimitValue] = useState<T>(value);

  useEffect(() => {
    latestValue.current = value;
  }, [value]);

  useEffect(() => {
    const timer = setInterval(() => {
      setRateLimitValue(latestValue.current);
    }, delay);

    return () => {
      clearInterval(timer);
    };
  }, [delay]);

  return rateLimitValue;
}

export default useRateLimit;
