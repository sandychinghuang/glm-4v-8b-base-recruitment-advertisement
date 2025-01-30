from pathlib import Path
from typing import Annotated, Union
import typer
from peft import PeftModelForCausalLM
from transformers import (
    AutoModel,
    AutoTokenizer,
)
import torch
from PIL import Image,ImageOps
import json
import json_repair


app = typer.Typer(pretty_exceptions_show_locals=False)


def load_model_and_tokenizer(
    model_dir: Union[str, Path], trust_remote_code: bool = True
):
    model_dir = Path(model_dir).expanduser().resolve()
    if (model_dir / "adapter_config.json").exists():
        import json

        with open(model_dir / "adapter_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
        model = AutoModel.from_pretrained(
            config.get("base_model_name_or_path"),
            trust_remote_code=trust_remote_code,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModelForCausalLM.from_pretrained(
            model=model,
            model_id=model_dir,
            trust_remote_code=trust_remote_code,
        )
        tokenizer_dir = model.peft_config["default"].base_model_name_or_path
    else:
        model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=trust_remote_code,
        encode_special_tokens=True,
        use_fast=False,
    )
    return model, tokenizer

def clean_generated_content(content):
    #清理生成的內容，移除多餘的字符、Markdown 標記和格式錯誤。
    # 移除 Markdown 標記和結尾的 <|endoftext|>
    content = content.replace("```json", "").replace("```", "").replace("<|endoftext|>", "").strip()
    # content = content.replace("<|endoftext|>", "").strip()

    # 移除 #zh-tw（如果出現）
    content = content.replace("#zh-tw", "").strip()
    
    return content

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
        
        print("現在計算第",idx+1)
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
            # 公司名稱
                if isinstance(generated, list):
                    print(f"第 {idx + 1} 的 generated 是列表，嘗試選取唯一對象") # 如果 generated 是列表，找到包含 "公司名稱" 的第一個有效對象
                    generated = next(
                        (item for item in generated if isinstance(item, dict) and "公司名稱" in item),
                        None
                    )
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
def main(
    model_dir: Annotated[str, typer.Argument(help="")],
):
    model, tokenizer = load_model_and_tokenizer(model_dir)
    # print(model)
    # print("======")
    # print(tokenizer)
    generate_kwargs = {
        "max_new_tokens": 2500,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }
    MAX_RESOLUTION = (1120, 1120)
    
######### 要test的文件路徑，文件格式請參見test.jsonl，xlsx轉換jsonl腳本:xlsx2jsonl.py ##########
    data_file = "test.jsonl"
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # 初始化結果列表
    predictions = []

    # 遍歷每條數據進行推論
    for entry in data:
        try:
            user_message = entry["messages"][0]
            assistant_label = entry["messages"][1]
            image_path = user_message["image"]
            query = """
        現在你是一位徵才資訊分類達人，請你詳細描述完整，包括廣告中出現的所有文字和訊息，沒有出現的請不要擅自加入，若無該欄位資請填'無'。 
        -請首先找到公司名稱、聯絡電話、公司地址。公司名稱為可能的公司名稱、店名、單位、廠房名稱、工作性質等，請寫完整，不要刪字，真的沒有才填'無'。每個聯絡電話只可包含電話(有分機請加在電話後面，例如'分機11')，若不同區域有不同電話，請特別註明區域與電話，勿加入與職位和工作無關的資訊，例如'誠徵'、'Line Id'、'意洽'、'鄭經理'、'陳小姐'、'孫先生'等。公司地址不一定是完整地址，只要可能為公司位置的地名都可寫，若出現括號，例如'高雄市…166號(舊市議會主婦商場)'時，只要地址的部分，括號不要。
        - 接下來，逐個描述每個職位。每個職位應包含：
        - 職位名稱(請寫完整，若一行多個職位請分開)
            -工作地點(和公司地址填一樣的，或完整特殊工作地點，同個職位多個地點(只有地點不一樣，其他內容都一樣的職位)用'、'隔開就好，特殊狀況:地點出現括號，例如'八德 (近八德區公所)'，只要地點的部分，括號不要，若都沒有請填'無')
            -工作開始與結束時間(用24小時制表示)
            -薪資上限與下限的注意事項:
            1.若是給只有定一個數字，例如'薪28000'、'時薪200'、'月薪32000元'、'2000'，則薪資上限和下限"都要"填上該數字。
            2.若遇到'時薪183元起'、'27470+獎金'、'1700以上'等情況，就填數字在'薪資下限'，'薪資上限'填'無'。
            -薪資方式(圖片中若出現'薪資面議'、'面議'等字詞屬於下一個步驟的'其他資訊'，此時”薪資上下限”和”薪資方式”填"無")。
            - 若每個職位的條件相似但時間不同，仍需單獨列出，並且剩餘資訊不可忽略。
            - 任何其他屬於該職位的資訊，(注意:不要更改原始敘述方式，例如:'/'；若有多項，請以'、'區隔)，勿加入與職位和工作無關的資訊，例如'誠徵'、'Line Id'、'意洽'、'鄭經理'、'陳小姐'、'孫先生'、'月入幾萬不是夢'、'財富自由'、'無誠勿擾'、'見報一週內有效'等。
        - 最後，請查看是否有適用於所有職位的其他資訊，如福利條件，並將此類資訊重複放在每個職位的'其他資訊'欄中，如果資訊位置靠近特定職位，請仔細推敲其是否只有適用於該職位。

        您的回應必須使用#zh-tw，並以正確的JSON格式呈現，如下所示：
        {
            “公司名稱”: “<公司名稱或'無'>”, 
            “聯絡電話”: [“<只可填電話(忽略人名)，若不同區域請特別註明區域與電話，或'無'>”],
            “公司地址”: “<公司地址或'無'>”,
            “職位”: [
                {
                    “職位名稱”: “<職位名稱>”,
                    “工作地點”: “<公司地址或特殊工作地點，若都沒有填'無'>”,
                    “工作開始時間”: “<用24小時制表示或'無'>”,
                    “工作結束時間”: “<用24小時制表示或'無'>”,
                    “薪資下限”: “<薪資下限或'無'>”,
                    “薪資上限”: “<薪資上限或'無'>”,
                    “薪資方式”: “<'時薪'、'日薪'、'月薪'或'無'>”,
                    “其他資訊”: “<任何其他屬於該職位的資訊(注意:不要更改原始敘述方式，例如:'/'；若有多項，請以'、'區隔)或'無'>”
                },
                …
            ],
        }
        全部都生成後，請重新檢查是否與原始內容完全一致，以及職位與工作地點是否需排列組合，以及其他資訊有沒有搞混特定職位專屬，或是所有職位都適用。
        """
            label = assistant_label["content"]
            if isinstance(label, str):
                try:
                    label = json.loads(label)  # 嘗試解析標注
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON in label for {image_path}")
                    predictions.append({"file_name": image_path, "generated": None, "label": None})
                    continue
            elif isinstance(label, dict):
                label = label
            else:
                print(f"Unexpected label format for {image_path}: {type(label)}")
                predictions.append({"file_name": image_path, "generated": None, "label": None})
                continue

            try:
                image = Image.open(image_path).convert('RGB')
                if image.size[0] > MAX_RESOLUTION[0] or image.size[1] > MAX_RESOLUTION[1]:
                    image = ImageOps.contain(image, MAX_RESOLUTION)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                predictions.append({"file_name": image_path, "generated": None, "label": label})
                continue
            
            max_retries = 10  # 最大重試次數
            retries = 0
            generated = None  # 初始化生成內容

            while retries < max_retries:
                try:
                    inputs = tokenizer.apply_chat_template(
                        [{"role": "user", "image": image, "content": query}],
                        add_generation_prompt=True, tokenize=True, return_tensors="pt",
                        return_dict=True
                    ).to(model.device)

                    with torch.no_grad():
                        outputs = model.generate(**inputs, **generate_kwargs)
                        outputs = outputs[:, inputs['input_ids'].shape[1]:]
                        generated_content = tokenizer.decode(outputs[0]).strip()

                        # 清理生成內容
                        cleaned_content = clean_generated_content(generated_content)

                        # 嘗試解析 JSON
                        generated = json.loads(cleaned_content)
                        break  # 成功解析，退出循環
                except json.JSONDecodeError as e:
                    retries += 1
                    print(f"Error parsing JSON for {image_path} on attempt {retries}: {e}")
                    print(f"Cleaned content for {image_path}: {cleaned_content}")
                    if retries >= max_retries:
                        print(f"Failed to generate valid JSON for {image_path}. Use json_repair to fix it.")
                        generated = json_repair.loads(cleaned_content)
                        # generated = None
                except Exception as e:
                    print(f"Error during model inference for {image_path}: {e}")
                    generated = None
                    break  # 如果是其他錯誤，退出循環

            # 添加到結果列表，包含file_name、generated、label
            predictions.append({"file_name": image_path, "generated": generated, "label": label})

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            predictions.append({"file_name": image_path, "generated": None, "label": label})
        
########## 保存生成內容的路徑 ########
        output_path = "glm_test_predictions.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for prediction in predictions:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

        print(f"Predictions saved to {output_path}")

    metrics = calculate_metrics(predictions)
    print(json.dumps(metrics, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    app()
