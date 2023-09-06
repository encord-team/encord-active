import { createContext, useContext, useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { useSessionStorage } from "usehooks-ts";

export const AuthContext = createContext<{ token: string | null }>({
  token: null,
});

export const useAuth = () => useContext(AuthContext);

export const createAuthContext = () => {
  const [searchParams, setSearchParams] = useSearchParams();

  const [sessionToken, setSessionToken] = useSessionStorage<string | null>(
    "token",
    null,
  );

  const queryToken = useMemo(() => searchParams.get("token"), []);

  useEffect(() => {
    if (!queryToken) return;
    setSessionToken(queryToken);
    searchParams.delete("token");
    setSearchParams(searchParams);
  }, [queryToken]);

  return { token: sessionToken ?? queryToken };
};
