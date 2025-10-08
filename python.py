import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests # Thư viện cần thiết để gọi Gemini API

# --- Cấu hình API và Thông số ---
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Phương Án Kinh Doanh",
    layout="wide"
)

st.title("Ứng dụng Đánh Giá Phương Án Kinh Doanh 📊")
st.markdown("Chuyên gia tài chính AI: Trích xuất, tính toán và phân tích dự án đầu tư.")

# Khởi tạo state (Quan trọng cho các thao tác lặp lại)
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'cash_flow_df' not in st.session_state:
    st.session_state.cash_flow_df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = None

# --- Hàm gọi API Gemini với Backoff ---
@st.cache_data(show_spinner=False)
def call_gemini_api(payload, api_key):
    """Thực hiện gọi API Gemini với cơ chế Exponential Backoff."""
    if not api_key:
        raise ValueError("Khóa API không được tìm thấy.")

    url = f"{API_URL_BASE}{payload.get('model', MODEL_NAME)}:generateContent?key={api_key}"
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Gửi yêu cầu
            response = requests.post(
                url, 
                headers={'Content-Type': 'application/json'}, 
                data=json.dumps(payload)
            )
            response.raise_for_status() # Ném lỗi cho mã trạng thái HTTP không thành công
            
            result = response.json()
            
            # Kiểm tra lỗi từ API
            if 'error' in result:
                st.error(f"Lỗi API từ Gemini: {result['error']['message']}")
                return None

            # Trích xuất văn bản/JSON
            candidate = result.get('candidates', [{}])[0]
            if candidate and candidate.get('content') and candidate['content'].get('parts'):
                return candidate['content']['parts'][0].get('text')

            return None
            
        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                sleep_time = 2 ** attempt # Exponential backoff
                time.sleep(sleep_time)
                continue
            st.error(f"Lỗi HTTP: {e}. Mã trạng thái: {response.status_code}")
            return None
        except Exception as e:
            st.error(f"Lỗi không xác định khi gọi API: {e}")
            return None
    st.error("Thử lại thất bại sau nhiều lần.")
    return None

# --- Chức năng 1: Trích xuất Dữ liệu bằng AI (Có cấu trúc) ---
def extract_parameters(document_text, api_key):
    """
    Sử dụng Gemini API với JSON Schema để trích xuất 6 thông số tài chính.
    """
    st.session_state.ai_analysis = None # Reset analysis
    st.session_state.cash_flow_df = None
    st.session_state.metrics = None
    
    # Định nghĩa Schema cho các thông số cần trích xuất
    extraction_schema = {
        "type": "OBJECT",
        "properties": {
            "InvestmentCapital": {"type": "NUMBER", "description": "Tổng vốn đầu tư ban đầu ($t_0$), đơn vị VND."},
            "ProjectLifespan": {"type": "INTEGER", "description": "Dòng đời dự án, số năm hoạt động (vd: 5, 10)."},
            "AnnualRevenue": {"type": "NUMBER", "description": "Doanh thu hàng năm, đơn vị VND. (Giả định ổn định)."},
            "AnnualCost": {"type": "NUMBER", "description": "Chi phí hoạt động hàng năm (chưa bao gồm Khấu hao, Thuế), đơn vị VND. (Giả định ổn định)."},
            "WACC": {"type": "NUMBER", "description": "Tỷ lệ chiết khấu (WACC) dưới dạng thập phân (ví dụ: 0.12 cho 12%)."},
            "TaxRate": {"type": "NUMBER", "description": "Thuế suất thuế thu nhập doanh nghiệp dưới dạng thập phân (ví dụ: 0.2 cho 20%)."}
        },
        "required": ["InvestmentCapital", "ProjectLifespan", "AnnualRevenue", "AnnualCost", "WACC", "TaxRate"]
    }

    # Prompt hướng dẫn AI trích xuất
    system_prompt = (
        "Bạn là một chuyên gia trích xuất dữ liệu tài chính. Nhiệm vụ của bạn là đọc nội dung dự án được cung cấp và trích xuất "
        "chính xác 6 thông số sau vào định dạng JSON bắt buộc. Bỏ qua tất cả văn bản khác. "
        "Đảm bảo các giá trị là số và tỷ lệ là số thập phân. Giả định dòng đời dự án không quá 10 năm."
    )
    
    user_query = f"Trích xuất các thông số tài chính cần thiết từ nội dung dự án sau:\n\n---\n{document_text}\n---"

    payload = {
        "model": MODEL_NAME,
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": extraction_schema
        }
    }

    with st.spinner("Đang sử dụng AI để lọc thông tin dự án..."):
        json_string = call_gemini_api(payload, api_key)
        
    if json_string:
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            st.error(f"Lỗi: AI trả về định dạng JSON không hợp lệ. Vui lòng kiểm tra lại nội dung file. Chi tiết lỗi: {e}")
            st.code(json_string) # Hiển thị JSON lỗi để debug
            return None
    return None

# --- Chức năng 2 & 3: Tính toán Dòng tiền & Chỉ số ---

# Khấu hao tuyến tính (Giả định giá trị thanh lý = 0)
def calculate_depreciation(investment, lifespan):
    """Tính Khấu hao theo phương pháp tuyến tính."""
    if lifespan > 0:
        return investment / lifespan
    return 0

@st.cache_data
def calculate_financials(data):
    """Xây dựng Bảng dòng tiền và tính toán các chỉ số đánh giá."""
    Inv = data['InvestmentCapital']
    L = data['ProjectLifespan']
    Rev = data['AnnualRevenue']
    Cost = data['AnnualCost']
    WACC = data['WACC']
    Tax = data['TaxRate']

    # 1. Bảng Dòng tiền (Cash Flow Table)
    years = list(range(1, L + 1))
    
    # Khấu hao
    Depr = calculate_depreciation(Inv, L)
    
    # Tính toán dòng tiền từng năm
    CF_data = []
    
    # Giai đoạn t=0 (Đầu tư)
    CF_data.append({
        'Năm': 0, 
        'Vốn đầu tư': -Inv, 
        'Doanh thu': 0, 
        'Chi phí (Trừ Khấu hao)': 0,
        'Lợi nhuận trước thuế': 0, 
        'Thuế phải nộp': 0, 
        'Dòng tiền thuần (CF)': -Inv, 
        'Dòng tiền chiết khấu (DCF)': -Inv,
        'Dòng tiền lũy kế (CCF)': -Inv,
        'Dòng tiền chiết khấu lũy kế (DCCF)': -Inv,
        'PV của CF (NPV)': 0
    })

    # Giai đoạn t=1 đến t=L
    for t in years:
        # Lợi nhuận trước Thuế và Lãi (EBIT)
        EBIT = Rev - Cost - Depr
        
        # Thuế
        Tax_paid = EBIT * Tax if EBIT > 0 else 0
        
        # Lợi nhuận sau thuế (Net Income)
        Net_Income = EBIT - Tax_paid
        
        # Dòng tiền thuần (CF = Net Income + Khấu hao)
        CF = Net_Income + Depr
        
        # Dòng tiền chiết khấu (Discounted Cash Flow - DCF)
        DCF = CF / (1 + WACC) ** t
        
        CF_data.append({
            'Năm': t, 
            'Vốn đầu tư': 0, 
            'Doanh thu': Rev, 
            'Chi phí (Trừ Khấu hao)': Cost + Depr,
            'Lợi nhuận trước thuế': EBIT, 
            'Thuế phải nộp': Tax_paid, 
            'Dòng tiền thuần (CF)': CF, 
            'Dòng tiền chiết khấu (DCF)': DCF,
            'Dòng tiền lũy kế (CCF)': 0, # Sẽ tính sau
            'Dòng tiền chiết khấu lũy kế (DCCF)': 0, # Sẽ tính sau
            'PV của CF (NPV)': 0
        })

    df = pd.DataFrame(CF_data)
    
    # Tính toán Lũy kế (CCF và DCCF)
    df['Dòng tiền lũy kế (CCF)'] = df['Dòng tiền thuần (CF)'].cumsum()
    df['Dòng tiền chiết khấu lũy kế (DCCF)'] = df['Dòng tiền chiết khấu (DCF)'].cumsum()

    # 2. Tính toán Chỉ số Đánh giá Hiệu quả
    
    # NPV (Net Present Value - Giá trị hiện tại thuần)
    # Dòng tiền bao gồm Inv ban đầu và CF thuần các năm sau
    npv_value = df['Dòng tiền chiết khấu (DCF)'].sum()
    
    # IRR (Internal Rate of Return - Tỷ suất sinh lời nội bộ)
    try:
        cf_for_irr = df['Dòng tiền thuần (CF)'].tolist()
        irr_value = np.irr(cf_for_irr)
    except Exception:
        irr_value = np.nan # Không thể tính nếu dòng tiền không đổi dấu

    # PP (Payback Period - Thời gian hoàn vốn)
    # Tìm năm mà CCF chuyển từ âm sang dương
    ccf = df['Dòng tiền lũy kế (CCF)'].values
    payback_year = np.argmax(ccf >= 0)
    
    if payback_year > 0:
        prev_year_cf = df.loc[payback_year - 1, 'Dòng tiền lũy kế (CCF)']
        current_year_cf = df.loc[payback_year, 'Dòng tiền thuần (CF)']
        pp_value = (payback_year - 1) + (abs(prev_year_cf) / current_year_cf)
    else:
        pp_value = L + 1 # Nếu không hoàn vốn trong dòng đời dự án
    
    # DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    # Tìm năm mà DCCF chuyển từ âm sang dương
    dccf = df['Dòng tiền chiết khấu lũy kế (DCCF)'].values
    discounted_payback_year = np.argmax(dccf >= 0)
    
    if discounted_payback_year > 0:
        prev_year_dcf = df.loc[discounted_payback_year - 1, 'Dòng tiền chiết khấu lũy kế (DCCF)']
        current_year_dcf = df.loc[discounted_payback_year, 'Dòng tiền chiết khấu (DCF)']
        dpp_value = (discounted_payback_year - 1) + (abs(prev_year_dcf) / current_year_dcf)
    else:
        dpp_value = L + 1

    metrics = {
        'NPV': npv_value,
        'IRR': irr_value,
        'PP': pp_value,
        'DPP': dpp_value,
        'WACC': WACC,
        'ProjectLifespan': L
    }
    
    return df, metrics

# --- Chức năng 4: Phân tích AI ---
def analyze_metrics(metrics, cash_flow_df, api_key):
    """Yêu cầu Gemini AI phân tích các chỉ số hiệu quả dự án."""
    
    metrics_str = "\n".join([f"- {k}: {v:,.2f}" for k, v in metrics.items() if k not in ['ProjectLifespan', 'WACC']])
    wacc_str = f"WACC (Tỷ lệ chiết khấu) được sử dụng là: {metrics['WACC'] * 100:.2f}%."
    lifespan_str = f"Dòng đời dự án là: {metrics['ProjectLifespan']} năm."
    
    prompt = f"""
    Bạn là một chuyên gia phân tích tài chính đầu tư. Hãy phân tích khách quan và chuyên sâu về hiệu quả của dự án dựa trên các chỉ số sau. 
    Đưa ra nhận định rõ ràng về tính khả thi (Khả thi/Không khả thi) và khuyến nghị đầu tư:

    1. **Các chỉ số đánh giá dự án:**
    {metrics_str}
    {wacc_str}
    {lifespan_str}

    2. **Yêu cầu phân tích:**
    - **NPV:** Đánh giá ý nghĩa của NPV (âm hay dương) so với mức đầu tư.
    - **IRR:** So sánh IRR với WACC (chi phí sử dụng vốn). Dự án có tạo ra giá trị không?
    - **PP & DPP:** Nhận xét về tốc độ hoàn vốn, đặc biệt so sánh PP và DPP để thấy rõ tác động của yếu tố thời gian và WACC.
    - **Kết luận chung:** Dự án này có nên được chấp thuận/đầu tư không? Giải thích ngắn gọn lý do.

    Bắt đầu bài phân tích của bạn với tiêu đề: "Phân Tích Hiệu Quả Đầu Tư Dự Án" và kết thúc bằng một kết luận rõ ràng.
    """
    
    payload = {
        "model": MODEL_NAME,
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": "Bạn là chuyên gia phân tích tài chính đầu tư."}]},
    }
    
    with st.spinner("Đang gửi chỉ số và chờ Gemini AI phân tích..."):
        return call_gemini_api(payload, api_key)


# --- Khu vực Xử lý File và Tương tác Người dùng ---

# Kiểm tra API Key
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.warning("⚠️ Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY' trong Streamlit Secrets. Vui lòng cấu hình Khóa API để sử dụng chức năng AI.")

col_upload, col_input = st.columns([2, 3])

# 1. Tải File và Text Input
with col_upload:
    uploaded_file = st.file_uploader(
        "1. Tải file Word/Text chứa phương án kinh doanh:",
        type=['docx', 'txt']
    )
    
    # Xử lý đọc file thành văn bản
    document_text = ""
    if uploaded_file is not None:
        try:
            # Đối với file docx, đọc thô có thể không hoàn hảo, nhưng Gemini có thể xử lý tốt hơn file txt
            document_text = uploaded_file.getvalue().decode("utf-8", errors='ignore')
            st.success("Đã đọc nội dung file.")
        except UnicodeDecodeError:
            st.warning("Không thể giải mã file Word phức tạp. Vui lòng dán nội dung vào ô bên cạnh.")
            document_text = uploaded_file.getvalue().decode('latin-1', errors='ignore')
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
            document_text = ""

with col_input:
    # Cho phép người dùng dán/chỉnh sửa nội dung
    text_input = st.text_area(
        "Hoặc dán/chỉnh sửa nội dung tài liệu (khuyến nghị nếu file Word phức tạp):",
        value=document_text,
        height=300
    )

if st.button("2. 🤖 Lọc Dữ liệu Tài chính bằng AI"):
    if not api_key:
        st.error("Vui lòng cấu hình Khóa API 'GEMINI_API_KEY' trước khi sử dụng chức năng AI.")
    elif not text_input.strip():
        st.error("Vui lòng tải lên file hoặc nhập nội dung dự án vào hộp văn bản.")
    else:
        extracted = extract_parameters(text_input, api_key)
        if extracted:
            st.session_state.extracted_data = extracted
            try:
                # Tính toán Dòng tiền và Chỉ số
                df, metrics = calculate_financials(extracted)
                st.session_state.cash_flow_df = df
                st.session_state.metrics = metrics
                st.success("Trích xuất và Tính toán thành công!")
            except Exception as e:
                st.session_state.extracted_data = None
                st.error(f"Lỗi khi tính toán tài chính: Vui lòng kiểm tra các tham số đã trích xuất. Chi tiết lỗi: {e}")
                st.warning("Đảm bảo các giá trị Doanh thu, Chi phí, Vốn đầu tư, và Dòng đời dự án là hợp lý.")


# --- Hiển thị Kết quả (Chức năng 2 & 3) ---

if st.session_state.extracted_data:
    data = st.session_state.extracted_data
    df = st.session_state.cash_flow_df
    metrics = st.session_state.metrics

    st.markdown("---")
    st.subheader("3. Các Thông số Dự án đã Trích xuất")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    col_p1.metric("💰 Vốn đầu tư ban đầu ($t=0$)", f"{data['InvestmentCapital']:,.0f} VND")
    col_p2.metric("🗓️ Dòng đời dự án", f"{data['ProjectLifespan']} năm")
    col_p3.metric("📈 WACC (Tỷ lệ chiết khấu)", f"{data['WACC'] * 100:.2f}%")
    
    col_p1.metric("💵 Doanh thu hàng năm", f"{data['AnnualRevenue']:,.0f} VND")
    col_p2.metric("💸 Chi phí hoạt động hàng năm", f"{data['AnnualCost']:,.0f} VND")
    col_p3.metric("🧾 Thuế suất (Tax)", f"{data['TaxRate'] * 100:.0f}%")

    st.markdown("---")
    st.subheader("4. Bảng Dòng tiền Dự án (Cash Flow Table)")

    if df is not None:
        # Định dạng hiển thị
        format_dict = {
            'Vốn đầu tư': '{:,.0f}',
            'Doanh thu': '{:,.0f}',
            'Chi phí (Trừ Khấu hao)': '{:,.0f}',
            'Lợi nhuận trước thuế': '{:,.0f}',
            'Thuế phải nộp': '{:,.0f}',
            'Dòng tiền thuần (CF)': '{:,.0f}',
            'Dòng tiền chiết khấu (DCF)': '{:,.0f}',
            'Dòng tiền lũy kế (CCF)': '{:,.0f}',
            'Dòng tiền chiết khấu lũy kế (DCCF)': '{:,.0f}',
            'PV của CF (NPV)': '{:,.0f}' # Chỉ dùng cột này để tính lũy kế bên trong hàm
        }

        # Áp dụng màu cho các cột lũy kế (quan trọng để xác định hoàn vốn)
        st.dataframe(
            df.style.format(format_dict).apply(
                lambda x: ['background-color: lightgreen' if v > 0 and x.name in ['Dòng tiền lũy kế (CCF)', 'Dòng tiền chiết khấu lũy kế (DCCF)'] else '' for v in x],
                axis=1
            ),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("5. Các Chỉ số Đánh giá Hiệu quả")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    if metrics:
        # NPV
        status_npv = "Dự án Khả thi" if metrics['NPV'] >= 0 else "Dự án Không khả thi"
        col_m1.metric("💵 NPV (Giá trị hiện tại thuần)", f"{metrics['NPV']:,.0f} VND", delta=status_npv)

        # IRR
        status_irr = "IRR > WACC" if metrics['IRR'] >= metrics['WACC'] else "IRR < WACC"
        col_m2.metric("📈 IRR (Tỷ suất sinh lời nội bộ)", f"{metrics['IRR'] * 100:.2f} %", delta=status_irr)

        # PP
        pp_delta = f"{metrics['PP'] - metrics['ProjectLifespan']:.2f} năm" if metrics['PP'] > metrics['ProjectLifespan'] else ""
        col_m3.metric("⏳ PP (Thời gian hoàn vốn)", f"{metrics['PP']:.2f} năm", delta=pp_delta)

        # DPP
        dpp_delta = f"{metrics['DPP'] - metrics['ProjectLifespan']:.2f} năm" if metrics['DPP'] > metrics['ProjectLifespan'] else ""
        col_m4.metric("📉 DPP (Hoàn vốn có chiết khấu)", f"{metrics['DPP']:.2f} năm", delta=dpp_delta)


    # --- Chức năng 4: Nhận xét AI ---
    st.markdown("---")
    st.subheader("6. 🧠 Phân tích Chuyên sâu (AI)")
    
    if st.button("Yêu cầu AI Phân tích Hiệu quả Dự án", key="analyze_button"):
        if not api_key:
            st.error("Vui lòng cấu hình Khóa API để sử dụng chức năng phân tích.")
        elif metrics:
            st.session_state.ai_analysis = analyze_metrics(metrics, df.to_markdown(), api_key)
            
    if st.session_state.ai_analysis:
        st.info(st.session_state.ai_analysis)
