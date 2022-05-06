import sys
import os
import pandas as pd
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import rawdata

if __name__=="__main__":
    pred_fn = "./data/prediction.parquet"
    transactions_parquet = "./data/test.parquet"
    vocab_dir = "vocabulary"
    predictions = pd.read_parquet(pred_fn)
    for col in predictions:
        predictions[col] = predictions[col].apply(lambda x: f"{x:010d}")
    customers = (pd.read_parquet(transactions_parquet,
            columns=["customer_id"])["customer_id"])
    predictions = predictions.values.tolist()
    for i in range(len(predictions)):
        predictions[i] = " ".join(predictions[i])
    predictions = pd.DataFrame({
        "customer_id": customers.values.tolist(),
        "prediction": predictions})
    out_fn = "submission.csv"
    predictions.to_csv(out_fn, index=False)

