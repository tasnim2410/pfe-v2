import {
  schemeTableau10,
  schemeSet3,
  schemePaired,
  schemeSet1,
  schemeSet2,
  schemeAccent,
  schemeDark2,
  schemePastel1,
  schemePastel2,
} from "d3-scale-chromatic";

const PALETTE: string[] = [
  ...schemeSet2,
  ...schemePastel1, 
  ...schemeSet3,
  ...schemeDark2,
  ...schemePaired,
  ...schemePastel2,
  ...schemeTableau10,
  ...schemeSet1,

  ...schemeAccent,

].flat();

export const COUNTRY_LIST: string[] = [
  "US","CN","WO","EP","KR","JP","ID","VN","BR","RU","NO","TH","TR","TW","DE","AT","AR","HU","NL","FI","MX","MA","SK","SA","PL","HK","IT","DK","PH","PE","SE","CO","MY","ES","PT","FR","AU","SG","EG","GR","IL","CA","VE","PK","IN","GB","IE","CH","BE","CL","BD","CZ","LU","NZ","ZA",
];

export const COUNTRY_COLOR_MAP: Record<string, string> = Object.fromEntries(
  COUNTRY_LIST.map((code, idx) => [code, PALETTE[idx % PALETTE.length]])
);

function hashColor(code: string): string {
  let h = 0 >>> 0;
  for (let i = 0; i < code.length; i++) {
    h = ((h * 31) + code.charCodeAt(i)) >>> 0;
  }
  return PALETTE[h % PALETTE.length];
}
export function colorForCountry(code: string): string {
  return COUNTRY_COLOR_MAP[code] ?? hashColor(code);
}
