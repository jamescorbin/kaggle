import sys
import os
import pands as pd
pt = os.path.abspath(os.path.join(
    __file__, os.pardir))
sys.path.insert(1, pt)
import rawdata

if __name__=="__main__":
    pred_fn = "prediction.parquet"
    transactions_parquet = "./data/transactions.parquet"
    vocab_dir = "vocabulary"
    prediction = pd.read_parquet(pred_fn)
    for col in prediction:
        prediction[col] = prediction[col].apply(lambda x: f"{x:010d}")
    transactions = pd.read_parquet(transactions_parquet)
    transactions = transactions.loc[transactions["test"]==1, "customer_id"]
    vocabulary = rawdata.load_vocabulary(vocab_dir)
    customer_id_map = {i + 1: x for i, x in vocabulary["customer_id"]}
    transactions["customer_id"] = (transactions["customer_id"]
            .apply(lambda x: customer_id_map[x]))
    predictions = predictions.values.tolist()
    for i in range(len(predictions)):
        predictions[i] = " ".join(predictions[i])
    predictions = pd.DataFrame({
        "customer_id": transactions["customer_id"],
        "prediction"; predictions})
    out_fn = "submission.csv"
    predictions.to_csv(out_fn, index=False)
