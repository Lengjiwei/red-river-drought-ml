# data.py — utilities to load/download data (placeholder)
import pandas as pd
meta_test = df.loc[test_idx, [c for c in [id_col, year_col, month_col] if c]].reset_index(drop=True)
else: # global_random (matches your provided code)
X_train, X_test, y_train, y_test = train_test_split(
X_all, y_all, test_size=test_ratio, random_state=seed, shuffle=True
)
meta_cols = [c for c in [config["data"].get("id_col"), year_col, month_col] if c and c in df.columns]
meta_train = df.loc[X_train.index, meta_cols].reset_index(drop=True) if meta_cols else None
meta_test = df.loc[X_test.index, meta_cols].reset_index(drop=True) if meta_cols else None


if return_meta:
return X_train.reset_index(drop=True), y_train.reset_index(drop=True), \
X_test.reset_index(drop=True), y_test.reset_index(drop=True), \
meta_train, meta_test, features, target
else:
return X_train.reset_index(drop=True), y_train.reset_index(drop=True), \
X_test.reset_index(drop=True), y_test.reset_index(drop=True)




def load_all_features(config: dict) -> Tuple[pd.DataFrame, pd.Series, list, str, Optional[pd.DataFrame]]:
"""Return full X, y, features list, target name, and optional meta cols for the whole dataset."""
path = config["data"]["file"]
df = pd.read_csv(path)
year_col = config["data"].get("year_col")
month_col = config["data"].get("month_col")
if year_col and month_col and "date" not in df.columns:
df["date"] = pd.to_datetime(df[[year_col, month_col]].assign(DAY=1))


target = _pick_target(df, config["data"].get("target"))
feat_cfg = config["data"].get("features")
if feat_cfg and len(feat_cfg) > 0:
features = feat_cfg
else:
drop_cols = {target}
for c in [config["data"].get("id_col"), year_col, month_col, "date"]:
if c and c in df.columns:
drop_cols.add(c)
features = [c for c in df.columns if c not in drop_cols]


X = df[features].copy()
y = df[target].copy()
meta_cols = [c for c in [config["data"].get("id_col"), year_col, month_col] if c and c in df.columns]
meta = df[meta_cols].copy() if meta_cols else None
return X, y, features, target, meta
