from embeddings.storage import load_embeddings, save_embeddings

EMBEDDINGS_PATH = "embeddings/embeddings.pkl"

person_name = input("Enter name to delete: ").strip()

database = load_embeddings(EMBEDDINGS_PATH)

if person_name in database:
    del database[person_name]

    save_embeddings(database, EMBEDDINGS_PATH)

    print(f"{person_name} deleted successfully.")
    print(f"Remaining identities: {len(database)}")
else:
    print(f"{person_name} not found.")