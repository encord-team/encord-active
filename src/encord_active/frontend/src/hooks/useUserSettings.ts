import { useLocalStorage } from "usehooks-ts";
import { UserSettings } from "../components/Types";

export function useUserSettings(): [
  settings: UserSettings,
  updateDisplaySettings: (
    key: keyof UserSettings,
    value: UserSettings[keyof UserSettings]
  ) => void
] {
  const DEFAULT_SETTINGS: UserSettings = {
    explorerGridCount: 5,
    explorerPageSize: 20,
  };

  const USER_SETTINGS_KEY = "userDisplaySettings";
  const [settings, updateSettings] = useLocalStorage<UserSettings>(
    USER_SETTINGS_KEY,
    DEFAULT_SETTINGS
  );

  function updateDisplaySettings<T>(
    key: keyof UserSettings,
    value: UserSettings[keyof UserSettings]
  ) {
    const newSettings = { ...settings, [key]: value };
    updateSettings(newSettings);
  }
  return [settings, updateDisplaySettings];
}
