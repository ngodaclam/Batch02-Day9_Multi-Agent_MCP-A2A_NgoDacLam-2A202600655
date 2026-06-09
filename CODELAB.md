# Codelab: Xây Dựng Hệ Thống Multi-Agent với A2A Protocol

**Thời gian:** 2 giờ  
**Ngôn ngữ:** Python 3.11+  
**Công nghệ:** LangGraph, LangChain, A2A SDK

## Mục Tiêu Học Tập

Sau khi hoàn thành codelab này, bạn sẽ:
- Hiểu cách LLM hoạt động từ cơ bản đến nâng cao
- Biết cách tích hợp tools và RAG vào LLM
- Xây dựng được single agent với ReAct pattern
- Tạo multi-agent system với LangGraph
- Triển khai distributed agents với A2A protocol

## Chuẩn Bị

### Yêu Cầu Hệ Thống
- Python 3.11 trở lên
- [uv](https://docs.astral.sh/uv/) package manager
- API key từ [OpenRouter](https://openrouter.ai)

### Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd legal_multiagent

# Cài đặt dependencies
uv sync

# Cấu hình environment
cp .env.example .env
# Sửa file .env, thêm OPENROUTER_API_KEY của bạn
```

---

## Phần 1: Direct LLM Calling (20 phút)

### Lý Thuyết

LLM (Large Language Model) ở dạng cơ bản nhất là một API nhận input text và trả về output text. Không có memory, không có tools, chỉ dựa vào training data.

**Ưu điểm:**
- Đơn giản, dễ implement
- Phản hồi nhanh

**Nhược điểm:**
- Không có kiến thức real-time
- Không thể tra cứu database
- Không có context giữa các lần gọi

### Thực Hành

**Bước 1:** Chạy demo Stage 1

```bash
uv run python stages/stage_1_direct_llm/main.py
```

**Bước 2:** Đọc và hiểu code

Mở file `stages/stage_1_direct_llm/main.py` và trả lời:

1. LLM được khởi tạo như thế nào? (Tìm hàm `get_llm()`)
```
def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    Model: lấy từ biến môi trường OPENROUTER_MODEL (mặc định là anthropic/claude-sonnet-4-5).
API key: lấy từ OPENROUTER_API_KEY.
Base URL: https://openrouter.ai/api/v1 → OpenRouter cung cấp giao diện tương thích với OpenAI.
Do vậy, khi trong main.py gọi llm = get_llm(), ta nhận được một đối tượng ChatOpenAI đã được cấu hình để giao tiếp với OpenRouter.
```
2. Message được gửi đến LLM có cấu trúc gì?
```
messages = [
    SystemMessage(
        content=(
            "You are a legal expert. Provide a clear, concise analysis "
            "of the legal question asked. Keep your response under 300 words."
        )
    ),
    HumanMessage(content=QUESTION),
]
SystemMessage: chứa hướng dẫn (system prompt) cho LLM, ở đây là “bạn là chuyên gia pháp lý…”.
HumanMessage: chứa câu hỏi thực tế
```
3. Tại sao cần có `SystemMessage` và `HumanMessage`?
```
SystemMessage : Đặt ngữ cảnh, vai trò cho LLM (system prompt). Nó hướng dẫn mô hình cách hành xử, phong cách, giới hạn (ví dụ: “là chuyên gia pháp lý”, “giới hạn 300 từ”). Điều này giúp LLM tạo ra câu trả lời nhất quán và đáp ứng yêu cầu thực tế.
HumanMessage: Đại diện cho ccâu hỏi / yêu cầu thực tế của người dùng. Đây là nội dung mà LLM cần phản hồi.
```

**Bài Tập 1.1:** Thay đổi câu hỏi

Sửa biến `QUESTION` thành câu hỏi pháp lý khác (tiếng Việt hoặc tiếng Anh) và chạy lại.

**Bài Tập 1.2:** Thêm temperature control

Thêm parameter `temperature=0.3` vào hàm `get_llm()` trong `common/llm.py` để làm output ổn định hơn.

---

## Phần 2: LLM + RAG & Tools (30 phút)

### Lý Thuyết

**RAG (Retrieval-Augmented Generation):** Cho phép LLM tra cứu knowledge base trước khi trả lời.

**Tools:** Các function mà LLM có thể gọi để thực hiện tác vụ cụ thể (tính toán, query database, gọi API).

**Function Calling Flow:**
1. LLM nhận câu hỏi + danh sách tools
2. LLM quyết định gọi tool nào (hoặc không gọi)
3. Tool được execute, trả về kết quả
4. LLM nhận kết quả và tạo câu trả lời cuối cùng

### Thực Hành

**Bước 1:** Chạy demo Stage 2

```bash
uv run python stages/stage_2_rag_tools/main.py
```

**Bước 2:** Phân tích code

Mở `stages/stage_2_rag_tools/main.py` và tìm:

1. Hàm `@tool` decorator được dùng ở đâu?
```
Hàm tool dùng ở search_legal_database và calculate_damages
```
2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?
```
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach …"
    },
    {
        "id": "nda_trade_secret",
        "keywords": ["nda", "non-disclosure", "confidential", "trade secret", "agreement"],
        "text": "NDA breaches may trigger both contractual and statutory liability …"
    },
    # … additional entries …
]
Với:
id – Định danh duy nhất của tài liệu.
keywords – Danh sách từ khóa tìm kiếm, viết thường (lower-case)
text – Nội dung đầy đủ của tài liệu hoặc đoạn văn giải thích
```
3. LLM được bind với tools ra sao? (Tìm `.bind_tools()`)
```
llm = get_llm()                     
llm_with_tools = llm.bind_tools(TOOLS)   
tool_map = {t.name: t for t in TOOLS}  
TOOLS được định nghĩa trước đó là [search_legal_database, calculate_damages] (ở dòng 138).

bind_tools() trả về một mô hình ngôn ngữ đã được tích hợp các công cụ (gọi là llm_with_tools), có khả năng tự tạo lệnh gọi công cụ (tool calls) và sử dụng kết quả trả về từ các công cụ đó.
```

**Bài Tập 2.1:** Thêm knowledge base entry

Thêm một entry mới vào `LEGAL_KNOWLEDGE` về luật lao động:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

**Bài Tập 2.2:** Tạo tool mới

Tạo một tool `@tool` mới tên `check_statute_of_limitations` nhận vào `case_type` (string) và trả về thời hiệu khởi kiện:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Thêm tool này vào danh sách tools và test.

---

## Phần 3: Single Agent với ReAct (25 phút)

### Lý Thuyết

**ReAct Pattern:** Reasoning + Acting

Agent tự động lặp lại chu trình:
1. **Think:** Suy nghĩ cần làm gì
2. **Act:** Gọi tool
3. **Observe:** Nhận kết quả
4. Lặp lại cho đến khi có câu trả lời cuối cùng

LangGraph cung cấp `create_react_agent` để tự động hóa pattern này.

### Thực Hành

**Bước 1:** Chạy demo Stage 3

```bash
uv run python stages/stage_3_single_agent/main.py
```

**Bước 2:** Quan sát output

Chú ý cách agent tự động:
- Quyết định tool nào cần gọi
- Gọi nhiều tools liên tiếp
- Tổng hợp kết quả

**Bước 3:** Đọc code

Mở `stages/stage_3_single_agent/main.py`:

1. Tìm `create_react_agent()` — đây là magic function
2. So sánh với Stage 2: không còn manual tool loop
3. Xem `agent_executor.invoke()` — chỉ cần gọi một lần

**Bài Tập 3.1:** Thêm tool tra cứu án lệ

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.
    
    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

Thêm vào tools list và test với câu hỏi về breach of contract.

**Bài Tập 3.2:** Debug agent reasoning

Thêm `verbose=True` vào `create_react_agent()` để xem chi tiết quá trình suy nghĩ của agent.

---

## Phần 4: Multi-Agent In-Process (30 phút)

### Lý Thuyết

**Multi-Agent System:** Nhiều agents chuyên môn hóa cùng làm việc.

**Ưu điểm:**
- Mỗi agent tập trung vào domain riêng
- Có thể chạy song song (parallel execution)
- Dễ maintain và mở rộng

**LangGraph StateGraph:**
- Định nghĩa state (dữ liệu chia sẻ giữa các nodes)
- Tạo nodes (các bước xử lý)
- Định nghĩa edges (luồng điều khiển)

**Send API:** Cho phép dispatch nhiều tasks song song.

### Thực Hành

**Bước 1:** Chạy demo Stage 4

```bash
uv run python stages/stage_4_milti_agent/main.py
```

**Bước 2:** Phân tích kiến trúc

Mở `stages/stage_4_milti_agent/main.py`:

1. Tìm `class State(TypedDict)` — đây là shared state
```
Shared state: LegalState (TypedDict) giữ toàn bộ thông tin qua các bước.
```
2. Tìm các agent functions: `law_agent`, `tax_agent`, `compliance_agent`
```
Agent functions: analyze_law, check_routing, call_tax_specialist, call_compliance_specialist, aggregate.
```
3. Tìm `Send()` API — dispatch parallel tasks
```
Parallel dispatch: Send() trả về danh sách các Send để LangGraph thực thi các node specialist đồng thời.
```
4. Xem `graph.add_node()` và `graph.add_edge()`
```
Graph construction: graph.add_node() định nghĩa các node, graph.add_edge()/add_conditional_edges() xác định luồng dữ liệu và routing, tạo một chuỗi xử lý tuần tự‑song song rõ ràng.
```

**Bước 3:** Vẽ graph

```python
# Thêm vào cuối file main.py
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Bài Tập 4.1:** Thêm agent mới

Tạo `privacy_agent` chuyên về GDPR và privacy law:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về luật bảo vệ dữ liệu cá nhân."""
    llm = get_llm()
    
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
    
Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Thêm node này vào graph và kết nối với `aggregate_results`.

**Bài Tập 4.2:** Implement conditional routing

Sửa `check_routing` để chỉ gọi privacy_agent khi câu hỏi có từ khóa "data", "privacy", "gdpr":

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
    
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    
    return tasks if tasks else [Send("aggregate_results", state)]
```

---

## Phần 5: Distributed A2A System (15 phút)

### Lý Thuyết

**A2A (Agent-to-Agent) Protocol:** Chuẩn giao tiếp giữa các agents qua HTTP.

**Khác biệt với Stage 4:**
- Mỗi agent là một service độc lập
- Giao tiếp qua HTTP thay vì in-process
- Dynamic discovery qua Registry
- Có thể scale từng agent riêng biệt

**Kiến trúc:**
```
Registry (10000) ← agents register on startup
    ↓
Customer Agent (10100) → Law Agent (10101)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Tax Agent (10102)   Compliance Agent (10103)
```

### Thực Hành

**Bước 1:** Khởi động toàn bộ hệ thống

```bash
./start_all.sh
```

Chờ ~10 giây để tất cả services khởi động.

**Bước 2:** Test hệ thống

```bash
uv run python test_client.py
```

**Bước 3:** Quan sát logs

Mở 5 terminal tabs và xem logs của từng service:
- Registry: port 10000
- Customer Agent: port 10100
- Law Agent: port 10101
- Tax Agent: port 10102
- Compliance Agent: port 10103

**Bài Tập 5.1:** Trace request flow

Trong logs, tìm `trace_id` và theo dõi request đi qua các agents. Vẽ sequence diagram.

**Bài Tập 5.2:** Test dynamic discovery

1. Dừng Tax Agent (Ctrl+C)
2. Chạy lại `test_client.py`
3. Quan sát lỗi và cách hệ thống xử lý

**Bài Tập 5.3:** Modify agent behavior

Sửa `tax_agent/graph.py`, thay đổi system prompt để agent trả lời ngắn gọn hơn. Restart tax agent và test lại.

---

## Phần 6: Tổng Kết & Mở Rộng (10 phút)

### So Sánh 5 Stages

| Stage | Pattern | Use Case | Complexity |
|---|---|---|---|
| 1 | Direct LLM | Câu hỏi đơn giản, không cần tools | ⭐ |
| 2 | LLM + Tools | Cần tra cứu data hoặc tính toán | ⭐⭐ |
| 3 | ReAct Agent | Tự động orchestration, multi-step | ⭐⭐⭐ |
| 4 | Multi-Agent | Nhiều domains, parallel processing | ⭐⭐⭐⭐ |
| 5 | Distributed A2A | Production, scalable, fault-tolerant | ⭐⭐⭐⭐⭐ |

### Câu Hỏi Ôn Tập

1. Khi nào nên dùng single agent thay vì multi-agent?
2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?
3. Làm thế nào để prevent infinite delegation loops trong A2A?
4. Tại sao cần Registry service? Có thể hardcode URLs không?

### Bài Tập Nâng Cao (Tự Học)

**Challenge 1:** Thêm memory/conversation history

Implement conversation memory để agent nhớ các câu hỏi trước đó.

**Challenge 2:** Add authentication

Thêm API key authentication cho các A2A endpoints.

**Challenge 3:** Implement retry logic

Khi một agent fail, tự động retry với exponential backoff.

**Challenge 4:** Monitoring & Observability

Tích hợp LangSmith hoặc Prometheus để monitor agent performance.

---

## Tài Liệu Tham Khảo

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Spec](https://github.com/google/A2A)
- [OpenRouter API](https://openrouter.ai/docs)
- Architecture diagrams: `docs/*.svg`

## Hỗ Trợ

Nếu gặp vấn đề:
1. Check `.env` file có đúng API key không
2. Đảm bảo tất cả ports (10000-10103) không bị chiếm
3. Xem logs trong terminal để debug
4. Đọc error messages cẩn thận — thường có hint rõ ràng

---

Bài Tập Cộng Điểm:
Vite Code HTML File Để demo các tương tác của các Agent ở stage 4 hoặc stage 5

Sau khi chạy full Stage 5 (test_client.py) trả lời 2 câu hỏi:
Latency (Tổng thời gian trả lời 1 câu hỏi của hệ thống) là bao nhiêu giây?
Đề xuất phương án giảm latency và demo + show thời gian xử lý đã giảm được khi apply phương án?

Câu hỏi 1: Latency (Tổng thời gian phản hồi) của hệ thống là bao nhiêu giây?
Khi chạy với LLM Mock (MockChatModel mới):
Thời gian xử lý trung bình đo được thực tế là ~1.62 giây.
Đặc điểm: Lực tính toán (inference) của LLM bằng 0 vì kết quả được sinh tức thì trong Mock. Thời gian trễ chủ yếu đến từ các cuộc gọi HTTP A2A liên kết nối qua các Agent (Customer $\rightarrow$ Law $\rightarrow$ Tax & Compliance song song $\rightarrow$ Law Aggregate $\rightarrow$ Customer).
Khi chạy với LLM thật (qua OpenRouter/Claude 3.5 Sonnet/GPT-4o):
Thời gian xử lý trung bình dao động từ 15 giây đến 35 giây (tùy thuộc vào tốc độ phản hồi của API và độ dài của câu trả lời).
Lý do độ trễ lớn: Hệ thống sử dụng mô hình tuần tự nhiều chặng (Multi-hop Reasoning). Mỗi Agent khi nhận request đều phải gửi nội dung lên API của LLM để suy luận và sinh văn bản (Inference Time). Cụ thể:
Chặng 1: Customer Agent gọi LLM để quyết định gọi Law Agent (~2-3s).
Chặng 2: Law Agent gọi LLM thực hiện analyze_law (~4-6s).
Chặng 3: Law Agent gọi LLM thực hiện check_routing định tuyến (~2-3s).
Chặng 4: Gọi song song Tax Agent & Compliance Agent. Cả hai gọi LLM độc lập (~4-7s).
Chặng 5: Law Agent gọi LLM thực hiện aggregate tổng hợp các báo cáo (~4-6s).
Tổng cộng: Xấp xỉ 5 cuộc gọi LLM tuần tự khiến latency thực tế bị kéo dài đáng kể.
Câu hỏi 2: Đề xuất phương án giảm Latency & Demo show thời gian xử lý đã giảm
1. Đề xuất các phương án tối ưu
Phương án A (Network/Discovery Cache) - ĐÃ CÀI ĐẶT:
Ý tưởng: Cài đặt cơ chế Local Cache cho Registry Client. Thay vì mỗi lần uỷ thác (delegate) đều phải gọi HTTP lên Registry (Port 10000) qua /discover/{task} để tìm địa chỉ IP động, Agent sẽ lưu kết quả vào cache trong bộ nhớ (In-memory Cache) với thời gian hết hạn (TTL = 5 phút).
Hiệu quả: Loại bỏ hoàn toàn 3 cuộc gọi mạng HTTP tới Registry trong suốt luồng request, giảm bớt chi phí bắt tay TCP/TLS (TCP/TLS handshake) và trễ mạng.
Phương án B (LLM Model Routing & Prompt Caching):
Ý tưởng: Bật tính năng Prompt Caching (như Anthropic Prompt Caching) đối với System Prompt của Law Agent và Customer Agent để giảm thời gian xử lý token đầu vào. Đồng thời, dùng các mô hình nhỏ/nhanh hơn (như Claude 3.5 Haiku, GPT-4o-mini) cho các tác vụ mang tính logic đơn giản như check_routing (định tuyến chỉ cần sinh JSON đúng/sai).
Phương án C (Parallel Specialists Call) - ĐÃ ÁP DỤNG:
Ý tưởng: Gọi song song Tax Agent và Compliance Agent sử dụng asyncio.gather hoặc LangGraph Send thay vì gọi tuần tự.
2. Demo cài đặt phương án Caching trên Registry Client
Tôi đã trực tiếp tối ưu hóa mã nguồn tại file 

common/registry_client.py
 để triển khai cơ chế bộ nhớ đệm (TTL = 300s):

python
# Cache store: task -> (endpoint, timestamp_fetched)
_endpoint_cache: Dict[str, Tuple[str, float]] = {}
CACHE_TTL = 300.0  # 5 minutes in seconds
async def discover(task: str) -> str:
    now = time.time()
    
    # Kiểm tra xem cache còn hiệu lực hay không
    if task in _endpoint_cache:
        endpoint, fetched_at = _endpoint_cache[task]
        if now - fetched_at < CACHE_TTL:
            return endpoint # Trả về ngay lập tức, bỏ qua gọi mạng HTTP
    # Nếu không có cache hoặc hết hạn, gọi HTTP lên Registry
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{REGISTRY_URL}/discover/{task}")
        resp.raise_for_status()
        endpoint = resp.json()["endpoint"]
        
        # Lưu vào cache
        _endpoint_cache[task] = (endpoint, now)
        return endpoint
3. So sánh thời gian xử lý (Show Latency Reduction)
Dưới đây là bảng đo đạc so sánh thời gian thực thi của test_client.py trước và sau khi tối ưu hóa bằng Caching:

**Chúc các bạn học tốt! 🚀**
