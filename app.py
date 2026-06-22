import json
import re
from datetime import datetime
from pathlib import Path

import gradio as gr

from scripts.inference_utils import NewsGenerator


PROJECT_ROOT = Path(__file__).resolve().parent
HISTORY_PATH = PROJECT_ROOT / "data" / "user_history.json"
EXPORT_DIR = PROJECT_ROOT / "exports"

generator = NewsGenerator()


EXAMPLE_NEWS = [
    [
        (
            "近日，教育部发布关于加强人工智能教育应用的相关通知，提出将进一步推动人工智能技术与高等教育、职业教育和基础教育深度融合。"
            "通知指出，各高校应结合自身学科优势，围绕智能计算、数据分析、机器人技术、智能制造、智慧医疗等方向优化课程体系，"
            "建设一批人工智能相关精品课程和实践平台。同时，鼓励高校与企业、科研院所开展合作，共同建设实训基地，"
            "推动学生在真实产业场景中提升工程实践能力。业内专家认为，人工智能正在成为推动新一轮科技革命和产业变革的重要力量，"
            "高校加快相关专业建设，有助于培养更多适应未来产业发展的复合型人才。"
        )
    ],
    [
        (
            "某市近日启动新一轮老旧小区改造工程，计划对全市多个建成时间较早、基础设施相对薄弱的小区进行综合提升。"
            "本次改造内容包括道路修缮、雨污管网更新、公共照明优化、电梯加装、停车位整治以及适老化设施建设等。"
            "当地住建部门表示，改造工作将充分听取居民意见，按照“一小区一方案”的原则推进，尽量减少施工对居民日常生活的影响。"
            "社区工作人员介绍，部分小区还将同步建设便民服务站和公共活动空间，为老人、儿童和上班族提供更加便利的生活环境。"
            "居民代表认为，老旧小区改造不仅改善居住条件，也有助于提升基层治理能力和城市公共服务水平。"
        )
    ],
    [
        (
            "今年以来，多地新能源汽车产业链企业加快扩产步伐，动力电池、智能座舱、汽车芯片和充电基础设施等领域投资持续增长。"
            "业内机构数据显示，随着国内消费需求回暖和海外市场订单增加，新能源汽车出口保持较快增长态势。"
            "多家整车企业表示，将继续加大研发投入，重点提升电池安全、续航能力、智能驾驶辅助和整车能耗管理水平。"
            "与此同时，地方政府也在推动充电网络建设和产业园区配套升级，吸引上下游企业集聚发展。"
            "专家指出，新能源汽车产业正在从规模扩张转向技术竞争和品牌竞争，供应链稳定性与核心技术自主能力将成为企业长期发展的关键。"
        )
    ],
]

HISTORY_HEADERS = ["生成时间", "生成标题", "摘要长度", "关键词数量", "标题覆盖率", "摘要覆盖率"]


def load_history():
    if not HISTORY_PATH.exists():
        return []

    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []

    return data if isinstance(data, list) else []


def save_history(records):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def get_user_records(username):
    username = (username or "").strip()
    if not username:
        return []

    return [record for record in load_history() if record.get("username") == username]


def format_history_table(records):
    rows = []
    for record in reversed(records):
        metrics = record.get("metrics", {})
        rows.append(
            [
                record.get("created_at", ""),
                record.get("title", ""),
                record.get("summary_length_option", ""),
                record.get("top_k", ""),
                f"{metrics.get('title_keyword_coverage', 0)}%",
                f"{metrics.get('summary_keyword_coverage', 0)}%",
            ]
        )
    return rows


def format_record_detail(record):
    if not record:
        return "暂无历史记录。"

    metrics = record.get("metrics", {})
    keywords = "、".join(record.get("keywords", []))
    return (
        f"生成时间：{record.get('created_at', '')}\n"
        f"用户：{record.get('username', '')}\n"
        f"摘要长度：{record.get('summary_length_option', '')}\n"
        f"关键词数量：{record.get('top_k', '')}\n\n"
        f"生成标题：\n{record.get('title', '')}\n\n"
        f"生成摘要：\n{record.get('summary', '')}\n\n"
        f"关键词：\n{keywords}\n\n"
        f"结果分析：\n"
        f"生成耗时：{metrics.get('time_cost', 0)} 秒\n"
        f"标题字数：{metrics.get('title_length', 0)} 字\n"
        f"摘要字数：{metrics.get('summary_length', 0)} 字\n"
        f"标题关键词覆盖率：{metrics.get('title_keyword_coverage', 0)}%\n"
        f"摘要关键词覆盖率：{metrics.get('summary_keyword_coverage', 0)}%\n\n"
        f"新闻正文：\n{record.get('input_text', '')}"
    )


def build_history_outputs(username):
    records = get_user_records(username)
    latest = records[-1] if records else None
    return format_history_table(records), format_record_detail(latest)


def login(username):
    username = (username or "").strip()
    if not username:
        return "", "请先输入用户名。", [], "暂无历史记录。"

    table, detail = build_history_outputs(username)
    return username, f"当前用户：{username}", table, detail


def build_analysis(result):
    return (
        f"生成耗时：{result['time_cost']} 秒\n"
        f"标题字数：{result['title_length']} 字\n"
        f"摘要字数：{result['summary_length']} 字\n"
        f"标题关键词覆盖率：{result['title_keyword_coverage']}%\n"
        f"摘要关键词覆盖率：{result['summary_keyword_coverage']}%"
    )


def generate_news(text, summary_length, top_k, username):
    username = (username or "").strip()
    text = (text or "").strip()

    if not username:
        return "请先输入用户名并登录。", "", "", "", "生成前请先完成用户名登录。", [], "暂无历史记录。"

    if not text:
        table, detail = build_history_outputs(username)
        return "请先输入新闻正文。", "", "", "", "本次没有生成记录。", table, detail

    result = generator.run(
        text=text,
        summary_length=summary_length,
        top_k=int(top_k),
    )

    keywords_text = "、".join(result["keywords"])
    analysis = build_analysis(result)

    record = {
        "username": username,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_text": text,
        "summary_length_option": summary_length,
        "top_k": int(top_k),
        "title": result["title"],
        "summary": result["summary"],
        "keywords": result["keywords"],
        "metrics": {
            "time_cost": result["time_cost"],
            "title_length": result["title_length"],
            "summary_length": result["summary_length"],
            "title_keyword_coverage": result["title_keyword_coverage"],
            "summary_keyword_coverage": result["summary_keyword_coverage"],
        },
    }

    records = load_history()
    records.append(record)
    save_history(records)

    table, detail = build_history_outputs(username)
    return result["title"], result["summary"], keywords_text, analysis, "生成完成，记录已保存。", table, detail


def refresh_history(username):
    username = (username or "").strip()
    if not username:
        return [], "请先输入用户名并登录。"

    return build_history_outputs(username)


def export_user_history(username):
    username = (username or "").strip()
    if not username:
        return None, "请先输入用户名并登录。"

    records = get_user_records(username)
    if not records:
        return None, "当前用户暂无可导出的历史记录。"

    safe_name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", username).strip("_") or "user"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_path = EXPORT_DIR / f"{safe_name}_history.json"
    payload = {
        "username": username,
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "record_count": len(records),
        "records": records,
    }

    with export_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(export_path), f"已导出 {len(records)} 条记录。"


HEADER_MD = """
# 基于预训练生成模型的中文新闻标题与摘要生成系统

本系统面向中文新闻正文，使用 Randeng-T5 预训练生成模型，采用标题生成与摘要生成统一建模的多任务 Text-to-Text 方法。
系统支持新闻标题生成、新闻摘要生成、关键词提取、生成结果分析与用户历史记录管理，适用于课程项目展示、实验演示和模型效果对比。
"""


SYSTEM_INFO_MD = """
## 🧪 模型与实验信息

| 项目 | 内容 |
| --- | --- |
| 基础模型 | IDEA-CCNL/Randeng-T5-77M-MultiTask-Chinese |
| 训练方式 | 标题生成与摘要生成统一建模为 Text-to-Text 多任务学习 |
| 任务提示词 | `生成标题：`、`生成摘要：` |
| 训练数据 | 标题生成 5000 条，摘要生成 5000 条 |
| 验证数据 | 标题生成 1000 条，摘要生成 1000 条 |
| 训练轮数 | 1 epoch |
| 模型文件大小 | 296M |

### 验证指标

| 指标 | 数值 |
| --- | ---: |
| BLEU | 0.1602 |
| ROUGE | 0.2837 |
| ROUGE-1 | 0.3458 |
| ROUGE-2 | 0.2290 |
| ROUGE-L | 0.3074 |

### 系统功能

- 新闻生成：输入中文新闻正文，一键生成标题、摘要、关键词与结果分析。
- 我的记录：按当前用户名查看历史生成记录、刷新记录、查看最近一条详情，并导出 JSON 文件。
- 系统说明：展示模型来源、训练设置、任务形式和验证指标，便于课程汇报截图与演示。
"""


CUSTOM_CSS = """
.gradio-container {
    max-width: 1180px !important;
}
#project-header {
    padding: 18px 0 8px;
}
#project-header h1 {
    margin-bottom: 10px;
}
#login-status {
    min-height: 40px;
}
"""


with gr.Blocks(
    title="基于预训练生成模型的中文新闻标题与摘要生成系统",
    css=CUSTOM_CSS,
) as demo:
    current_user = gr.State("")

    gr.Markdown(HEADER_MD, elem_id="project-header")

    with gr.Row():
        username_input = gr.Textbox(
            label="用户名",
            placeholder="请输入用户名，用于保存和查看个人生成记录。",
            scale=4,
        )
        login_btn = gr.Button("登录", variant="primary", scale=1)
        login_status = gr.Textbox(label="登录状态", value="未登录", interactive=False, elem_id="login-status", scale=3)

    with gr.Tabs():
        with gr.Tab("新闻生成"):
            with gr.Row():
                with gr.Column(scale=5):
                    news_input = gr.Textbox(
                        label="新闻正文输入",
                        value=EXAMPLE_NEWS[0][0],
                        lines=12,
                        placeholder="请粘贴或输入中文新闻正文。",
                    )

                    with gr.Row():
                        summary_length = gr.Dropdown(
                            choices=["短", "中", "长"],
                            value="中",
                            label="摘要长度",
                        )
                        top_k = gr.Slider(
                            minimum=3,
                            maximum=12,
                            value=8,
                            step=1,
                            label="关键词数量",
                        )

                    generate_btn = gr.Button("一键生成标题与摘要", variant="primary")
                    generate_status = gr.Textbox(label="生成状态", value="等待输入。", interactive=False)

                    gr.Markdown("### 示例新闻")
                    gr.Examples(
                        examples=EXAMPLE_NEWS,
                        inputs=news_input,
                        label="点击示例自动填充新闻正文",
                    )

                with gr.Column(scale=5):
                    title_output = gr.Textbox(
                        label="📝 生成标题",
                        lines=2,
                        interactive=False,
                    )
                    summary_output = gr.Textbox(
                        label="📌 生成摘要",
                        lines=7,
                        interactive=False,
                    )
                    keywords_output = gr.Textbox(
                        label="🔑 关键词",
                        lines=2,
                        interactive=False,
                        placeholder="关键词将以中文顿号分隔显示。",
                    )
                    analysis_output = gr.Textbox(
                        label="📊 结果分析",
                        lines=7,
                        interactive=False,
                    )

        with gr.Tab("我的记录"):
            with gr.Row():
                refresh_btn = gr.Button("刷新我的记录", variant="primary")
                export_btn = gr.Button("导出当前用户记录为 JSON")

            history_table = gr.Dataframe(
                headers=HISTORY_HEADERS,
                datatype=["str", "str", "str", "number", "str", "str"],
                label="我的历史记录",
                interactive=False,
                wrap=True,
            )
            latest_detail = gr.Textbox(
                label="最近一条详情",
                value="暂无历史记录。",
                lines=16,
                interactive=False,
            )
            export_file = gr.File(label="导出文件")
            export_status = gr.Textbox(label="导出状态", interactive=False)

        with gr.Tab("系统说明"):
            gr.Markdown(SYSTEM_INFO_MD)

    login_btn.click(
        fn=login,
        inputs=username_input,
        outputs=[current_user, login_status, history_table, latest_detail],
        api_name=False,
        show_api=False,
    )

    username_input.submit(
        fn=login,
        inputs=username_input,
        outputs=[current_user, login_status, history_table, latest_detail],
        api_name=False,
        show_api=False,
    )

    generate_btn.click(
        fn=generate_news,
        inputs=[news_input, summary_length, top_k, current_user],
        outputs=[
            title_output,
            summary_output,
            keywords_output,
            analysis_output,
            generate_status,
            history_table,
            latest_detail,
        ],
        api_name=False,
        show_api=False,
    )

    refresh_btn.click(
        fn=refresh_history,
        inputs=current_user,
        outputs=[history_table, latest_detail],
        api_name=False,
        show_api=False,
    )

    export_btn.click(
        fn=export_user_history,
        inputs=current_user,
        outputs=[export_file, export_status],
        api_name=False,
        show_api=False,
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)
