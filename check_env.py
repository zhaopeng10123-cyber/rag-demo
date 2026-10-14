"""检查 LLM 和 Embedding 是否都能调通：python check_env.py"""
import config


def main() -> None:
    vec = config.get_embeddings().embed_query("测试")
    print(f"[Embedding] OK  模型 {config.EMBEDDING_MODEL}，维度 {len(vec)}")

    try:
        llm = config.get_llm()
        reply = llm.invoke("用一句话介绍你自己").content
        print(f"[LLM] OK  {llm.model_name}：{reply}")
    except Exception as e:
        print(f"[LLM] 失败：{e}")


if __name__ == "__main__":
    main()
