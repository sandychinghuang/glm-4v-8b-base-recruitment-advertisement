import json
from pathlib import Path
from typing import List
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):  # 如果是列表，递归展开
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)  # 如果是非列表，直接添加
    return flat_list

def flatten_if_needed(data):
    if isinstance(data, list) and any(isinstance(i, list) for i in data):
        # 如果列表中存在子列表，则展平
        return flatten(data)
    elif isinstance(data, str):
        # 如果是字符串，包装成列表
        return [data]
    return data  # 如果是一维列表，直接返回

def calculate_metrics(predictions):
    scores = {
        "公司名稱": {"EM": 0, "total": 0},
        "聯絡電話": {"precision": 0, "recall": 0, "total_label": 0, "total_generated": 0},
        "公司地址": {"EM": 0, "total": 0},
        "職位名稱": {"EM": 0, "total": 0},
        "工作地點": {"EM": 0, "total": 0},
        "工作開始時間": {"EM": 0, "total": 0},
        "工作結束時間": {"EM": 0, "total": 0},
        "薪資方式": {"correct": 0, "total": 0},
        "薪資下限": {"EM": 0, "total": 0},
        "薪資上限": {"EM": 0, "total": 0},
    }
    
    idx=0
    for pred in predictions:
        
        print("現在處理第",idx+1)
        label = pred["label"]
        generated = pred["generated"]

        try:
            scores_temp = {key: value.copy() for key, value in scores.items()}
            
            if generated is None:
                # 更新公司名稱和地址的總數量
                print("第",idx+1,"生成不對")
                
                scores["公司名稱"]["total"] += 1
                scores["公司地址"]["total"] += 1

                # 處理聯絡電話（僅有 label 時）
                label_phones = set(label.get("聯絡電話", []))
                scores["聯絡電話"]["total_label"] += len(label_phones)

                # 若職位資訊存在，更新對應字段的總數量
                for position in label["職位"]:
                    for field in ["職位名稱", "工作地點", "工作開始時間", "工作結束時間", "薪資方式", "薪資下限", "薪資上限"]:
                        scores[field]["total"] += 1
                        
            else:
                if isinstance(generated, list):
                    print(f"第 {idx + 1} 的 generated 是列表，嘗試選取唯一對象") # 如果 generated 是列表，找到包含 "公司名稱" 的第一個有效對象
                    generated = next(
                        (item for item in generated if isinstance(item, dict) and "公司名稱" in item),
                        None
                    )
                # 公司名稱
                scores["公司名稱"]["total"] += 1
                if label.get("公司名稱") == generated.get("公司名稱"):
                    scores["公司名稱"]["EM"] += 1

                label_phones = set(label.get("聯絡電話", []))
                generated_phones = set(flatten_if_needed(generated.get("聯絡電話", [])))
                scores["聯絡電話"]["precision"] += len(label_phones & generated_phones)
                scores["聯絡電話"]["recall"] += len(label_phones & generated_phones)
                scores["聯絡電話"]["total_label"] += len(label_phones)
                scores["聯絡電話"]["total_generated"] += len(generated_phones)
        
                # 公司地址
                scores["公司地址"]["total"] += 1
                if label.get("公司地址") == generated.get("公司地址"):
                    scores["公司地址"]["EM"] += 1

                # 職位相關字段
                label_positions = label.get("職位", [])
                generated_positions = generated.get("職位", [])
            
                for label_pos, generated_pos in zip(label_positions, generated_positions):
                    for field in ["職位名稱", "工作地點", "工作開始時間", "工作結束時間"]:
                        scores[field]["total"] += 1
                        if label_pos.get(field) == generated_pos.get(field):
                            scores[field]["EM"] += 1

                    # 薪資方式
                    scores["薪資方式"]["total"] += 1
                    if label_pos.get("薪資方式") == generated_pos.get("薪資方式"):
                        scores["薪資方式"]["correct"] += 1

                    # 薪資上下限
                    for salary_field in ["薪資下限", "薪資上限"]:
                        scores[salary_field]["total"] += 1
                        if label_pos.get(salary_field) == generated_pos.get(salary_field):
                            scores[salary_field]["EM"] += 1
            idx += 1
        except Exception as e:
            scores = scores_temp
            # print(f"Error processing prediction {idx}: {e}")
            print("第",idx+1,"無法識別")
            scores["公司名稱"]["total"] += 1
            scores["公司地址"]["total"] += 1

            # 處理聯絡電話（僅有 label 時）
            label_phones = set(label.get("聯絡電話", []))
            scores["聯絡電話"]["total_label"] += len(label_phones)

            # 若職位資訊存在，更新對應字段的總數量
            for position in label["職位"]:
                for field in ["職位名稱", "工作地點", "工作開始時間", "工作結束時間", "薪資方式", "薪資下限", "薪資上限"]:
                    scores[field]["total"] += 1
            idx += 1
            continue  # 繼續處理下一條數據

    # 計算結果
    results = {}
    for field, score in scores.items():
        if "EM" in score:
            results[field] = score["EM"] / score["total"] if score["total"] > 0 else 0
        elif field == "聯絡電話":
            precision = score["precision"] / score["total_generated"] if score["total_generated"] > 0 else 0
            recall = score["recall"] / score["total_label"] if score["total_label"] > 0 else 0
            results[field] = {"precision": precision, "recall": recall}
        elif field == "薪資方式":
            results[field] = score["correct"] / score["total"] if score["total"] > 0 else 0
    
    print("分數計算公司數量:",scores["公司名稱"]["total"])
    
    top_level_accuracy = (scores["公司名稱"]["EM"] + scores["公司地址"]["EM"]) / (
    scores["公司名稱"]["total"] + scores["公司地址"]["total"])
    job_level_accuracy = (
        scores["職位名稱"]["EM"]
        + scores["工作地點"]["EM"]
        + scores["工作開始時間"]["EM"]
        + scores["工作結束時間"]["EM"]
        + scores["薪資方式"]["correct"]
        + scores["薪資下限"]["EM"]
        + scores["薪資上限"]["EM"]
    ) / (
        scores["職位名稱"]["total"]
        + scores["工作地點"]["total"]
        + scores["工作開始時間"]["total"]
        + scores["工作結束時間"]["total"]
        + scores["薪資方式"]["total"]
        + scores["薪資下限"]["total"]
        + scores["薪資上限"]["total"]
    )
    # 整體準確率
    overall_accuracy = (
        scores["公司名稱"]["EM"] 
        + scores["公司地址"]["EM"]
        + scores["職位名稱"]["EM"]
        + scores["工作地點"]["EM"]
        + scores["工作開始時間"]["EM"]
        + scores["工作結束時間"]["EM"]
        + scores["薪資方式"]["correct"]
        + scores["薪資下限"]["EM"]
        + scores["薪資上限"]["EM"]
    ) / (
        scores["公司名稱"]["total"] 
        + scores["公司地址"]["total"]
        + scores["職位名稱"]["total"]
        + scores["工作地點"]["total"]
        + scores["工作開始時間"]["total"]
        + scores["工作結束時間"]["total"]
        + scores["薪資方式"]["total"]
        + scores["薪資下限"]["total"]
        + scores["薪資上限"]["total"]
    )
    results["top_level_accuracy"] = top_level_accuracy
    results["job_level_accuracy"] = job_level_accuracy
    results["overall_accuracy"] = overall_accuracy

    formatted_results = {
        "field_accuracy": {
            "公司名稱": round(results.get("公司名稱", 0), 4),
            "公司地址": round(results.get("公司地址", 0), 4),
            "聯絡電話 (precision)": round(results.get("聯絡電話", {}).get("precision", 0), 4),
            "聯絡電話 (recall)": round(results.get("聯絡電話", {}).get("recall", 0), 4),
            "職位名稱": round(results.get("職位名稱", 0), 4),
            "工作地點": round(results.get("工作地點", 0), 4),
            "工作開始時間": round(results.get("工作開始時間", 0), 4),
            "工作結束時間": round(results.get("工作結束時間", 0), 4),
            "薪資方式": round(results.get("薪資方式", 0), 4),
            "薪資下限": round(results.get("薪資下限", 0), 4),
            "薪資上限": round(results.get("薪資上限", 0), 4),
        },
        "top_level_accuracy": round(top_level_accuracy, 4),
        "job_level_accuracy": round(job_level_accuracy, 4),
        "overall_accuracy": round(overall_accuracy, 4),
    }
    # return results
    return formatted_results

@app.command()
def main(predictions_file: str):

    predictions_path = Path(predictions_file).resolve()

    if not predictions_path.exists():
        print(f"File not found: {predictions_file}")
        return

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]

    metrics = calculate_metrics(predictions)
    print(json.dumps(metrics, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    app()
