// savedChartsState.ts (Recoil example)
import { atom } from "recoil";
export type SavedChart = {
  id: string;
  title: string;
  ref: React.MutableRefObject<null>;
};

export const savedChartsState = atom<SavedChart[]>({
  key: "savedChartsState",
  default: [],
});
