import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer, util

# Step 1: Load and extract text from PDF
def extract_faq_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text

# Step 2: Parse questions and answers
def parse_faq(text):
    faq_pairs = re.findall(r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)", text, re.DOTALL)
    return [(q.strip(), a.strip()) for q, a in faq_pairs]

# Step 3: Chatbot class using embeddings
class EmbeddingFAQChatbot:
    def __init__(self, faq_data):  # ✅ Fixed __init__
        self.questions = [q for q, _ in faq_data]
        self.answers = [a for _, a in faq_data]
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # ✅ Load embedding model
        self.embeddings = self.model.encode(self.questions, convert_to_tensor=True)

    def get_response(self, user_input):
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(user_embedding, self.embeddings)[0]
        max_score, max_index = float(cosine_scores.max()), int(cosine_scores.argmax())

        if max_score < 0.4:
            return "Sorry, I didn't understand that. Could you rephrase it?"
        return self.answers[max_index]

# Main Execution
if __name__ == "__main__":  # ✅ Fixed __main__
    pdf_path = "C:\\Users\\admin\\OneDrive\\Desktop\\DVerse Project\\tech_support_faq.pdf"  # ✅ Correct path
    raw_text = extract_faq_from_pdf(pdf_path)
    faq_data = parse_faq(raw_text)

    chatbot = EmbeddingFAQChatbot(faq_data)

    print("🤖 Semantic FAQ Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", chatbot.get_response(user_input))
