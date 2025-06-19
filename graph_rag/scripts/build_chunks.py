import json
from graph_rag.core.pdf_parser import extract_texts_from_folder
from graph_rag.core.chunking import process_book, split_into_chunks

def main():
    pdf_folder = "data\\pdf_initial"
    output_path = "data\\chunks.json"

    texts = extract_texts_from_folder(pdf_folder)
    all_chunks = []

    for doc_id, full_text in enumerate(texts):
        full_text = process_book(full_text)
        chunks = split_into_chunks(full_text, chunk_size=200, overlap=50)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"doc{doc_id}_chunk{i}",
                "text": chunk
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()