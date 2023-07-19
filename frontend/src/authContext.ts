import { createContext, useContext } from "react";
import { useSearchParams } from "react-router-dom";

export const AuthContext = createContext<{ token: string | null }>({
  token: null,
});

export const useAuth = () => useContext(AuthContext);

export const createAuthContext = () => {
  const [searchParams] = useSearchParams();

  return { token: searchParams.get("token") };
};
