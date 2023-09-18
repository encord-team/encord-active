import { create } from "zustand";

interface DisplaySettings {
  showAnnotations: boolean;
  toggleShowAnnotations: () => void;
}

export const useDisplaySettings = create<DisplaySettings>()((set) => ({
  showAnnotations: true,
  toggleShowAnnotations: () =>
    set((prev) => ({ ...prev, showAnnotations: !prev.showAnnotations })),
}));
