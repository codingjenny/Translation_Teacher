import os
import gradio as gr
from openai import OpenAI

# api_key 已經設置成環境變數，並從環境變數取得 api_key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# 設定 api_key，讓 client 這個變數知道是在使用哪個 api
client = OpenAI(api_key=api_key)

# 測試 openai_api
# 定義生成翻譯題目的函數
def generate_ielts_question(vocabulary):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                { 
                    "role": "system",
                    "content": "You are a professional IELTS teacher. You are tasked to generate a IELTS translation question based on the given vocabulary. Your generated question should be in Traditional Chinese." 
                },
                {
                    "role": "user",
                    "content": f"Please generate a sentence in Traditional Chinese based on the following English vocabulary: {vocabulary}. Only generate the sentense, do not generate the answer.",
                }
            ],
            model="gpt-4o-mini",
            max_tokens = 50
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

def grade_ielts_answer(question, user_answer):
    try:
        grading_completion = client.chat.completions.create(
            messages=[
                { 
                    "role": "system",
                    "content": "You are a professional IELTS grader. Grade the user's answer to the translation question and provide feedback in Traditional Chinese." 
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nUser's answer: {user_answer}\n\nPlease grade the answer and provide detailed feedback in Traditional Chinese. Include the correct translation in your feedback."
                }
            ],
            model="gpt-4o-mini",
            max_tokens=500
        )
        return grading_completion.choices[0].message.content
    except Exception as e:
        return f"發生錯誤：{e}"

def ielts_workflow(vocabulary, user_answer, stored_question):
    if not stored_question:
        question = generate_ielts_question(vocabulary)
        return question, question, "", gr.update(visible=True), gr.update(visible=False)
    else:
        feedback = grade_ielts_answer(stored_question, user_answer)
        return stored_question, stored_question, feedback

with gr.Blocks() as demo:
    gr.Markdown("# IELTS 翻譯問題生成器和評分系統")
    
    with gr.Row():
        vocab_input = gr.Textbox(label="Input English Vocabulary")
        generate_btn = gr.Button("Generate Question")
    
    question_output = gr.Textbox(label="Generated Question", interactive=False)
    
    with gr.Row():
        answer_input = gr.Textbox(label="Your Translation Answer")
        submit_btn = gr.Button("Submit Answer")
    
    feedback_output = gr.Textbox(label="Score and Feedback")
    
    with gr.Row():
        retry_btn = gr.Button("Reanswer This Question")
        clear_all_btn = gr.Button("Clear All")
    
    question_state = gr.State("")

    def generate_question(vocab):
        question = generate_ielts_question(vocab)
        return question, question, "", gr.update(value="")

    def submit_answer(question, answer):
        feedback = grade_ielts_answer(question, answer)
        return feedback

    def clear_answer():
        return "", ""  # 返回兩個空字符串，分別用於清空答案輸入框和反饋輸出框

    def clear_all():
        return "", "", "", "", ""  # 清空所有輸入和輸出字段

    generate_btn.click(generate_question, 
                       inputs=[vocab_input], 
                       outputs=[question_output, question_state, feedback_output, answer_input])
    
    submit_btn.click(submit_answer, 
                     inputs=[question_state, answer_input], 
                     outputs=[feedback_output])
    
    retry_btn.click(clear_answer, 
                    outputs=[answer_input, feedback_output])
    
    clear_all_btn.click(clear_all, 
                        outputs=[vocab_input, question_output, question_state, answer_input, feedback_output])

demo.launch(share=True)
