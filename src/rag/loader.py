from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents():
    with open("docs/company_info.txt", "r") as f:
        content = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_text(content)
