import { create } from "zustand";

interface DispalySettings {
  showAnnotations: boolean;
  toggleShowAnnotations: () => void;
}

export const useDisplaySettings = create<DispalySettings>()((set) => ({
  showAnnotations: true,
  toggleShowAnnotations: () =>
    set((prev) => ({ ...prev, showAnnotations: !prev.showAnnotations })),
}));
