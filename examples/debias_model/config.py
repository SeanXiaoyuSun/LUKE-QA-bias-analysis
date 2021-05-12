class BertConfig(object):
    model_type = "bert"

    vocab_size=50000
    hidden_size=144
    num_hidden_layers=12
    num_attention_heads=12
    intermediate_size=576
    hidden_act="gelu"
    hidden_dropout_prob=0.1
    attention_probs_dropout_prob=0.1
    max_position_embeddings=512
    type_vocab_size=2
    initializer_range=0.02
    layer_norm_eps=1e-12
    pad_token_id=0
    gradient_checkpointing=False
    position_embedding_type="absolute"
    use_cache=True
 
       