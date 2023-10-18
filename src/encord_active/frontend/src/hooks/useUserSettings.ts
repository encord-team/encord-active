import { useLocalStorage } from "usehooks-ts";
import { UserSettings } from "../components/Types";

export function useUserSettings(): [
  settings: UserSettings,
  updateDisplaySettings: (settings: Partial<UserSettings>) => void
] {
  const DEFAULT_SETTINGS: UserSettings = {
    explorerGridCount: 5,
    explorerPageSize: 10,
  };

  const USER_SETTINGS_KEY = "userDisplaySettings";
  const [settings, updateSettings] = useLocalStorage<UserSettings>(
    USER_SETTINGS_KEY,
    DEFAULT_SETTINGS
  );

  function updateDisplaySettings(settingsToUpdate: Partial<UserSettings>) {
    const newSettings = { ...settings, ...settingsToUpdate };
    updateSettings(newSettings);
  }
  return [settings, updateDisplaySettings];
}
