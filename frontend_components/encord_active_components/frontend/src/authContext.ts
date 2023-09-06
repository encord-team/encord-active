import { createContext, useContext, useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { useSessionStorage } from "usehooks-ts";

export type AuthContextType = {
  token: string | null;
}

export const AuthContext = createContext<AuthContextType>({
  token: null,
});

export function useAuth(): AuthContextType {
  return useContext(AuthContext);
}

export function useCreateAuthContext(): AuthContextType {
  const [searchParams, setSearchParams] = useSearchParams();

  const [sessionToken, setSessionToken] = useSessionStorage<string | null>(
    "token",
    null
  );

  const queryToken = useMemo(() => searchParams.get("token"), []);

  useEffect(() => {
    if (!queryToken) {return;}
    setSessionToken(queryToken);
    searchParams.delete("token");
    setSearchParams(searchParams);
  }, [queryToken]);

  return { token: sessionToken ?? queryToken };
}
