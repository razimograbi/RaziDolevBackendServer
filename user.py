from fastapi import WebSocket

LANGUAGE_MAPPER = {
    "english": "en",
    "spanish" : "es",
    "french" : "fr",
    "german" : "de",
    "italian": "it",
    "portuguese" : "pt",
    "polish" : "pl",
    "turkish" : "tr",
    "russian":"ru",
    "dutch" : "nl",
    "Czech" : "cs",
    "arabic": "ar",
    "chinese" : "zh-cn",
    "japanese": "ja",
    "hungarian":"hu",
    "korean":"ko",
    "hindi" : "hi"

}


class User:
    def __init__(self, websocket: WebSocket, user_id: str, language: str, profile_name: str, email: str, embedding,
                 gpt_cond_latent):
        self.websocket = websocket
        self.user_id = user_id
        self.profile_name = profile_name or "Unknown"
        self.email = email or "No Email"
        self.embedding = embedding
        self.gpt_cond_latent = gpt_cond_latent

        language = language.lower()

        if language in LANGUAGE_MAPPER:
            self.language = LANGUAGE_MAPPER[language]
            self.is_xtts_supported = True
        else:
            self.language = language
            self.is_xtts_supported = False

    def print_user_data(self) -> str:
        details = (f"WebSocket: {self.websocket}, "
                   f"User ID: {self.user_id}, "
                   f"Profile Name: {self.profile_name}, "
                   f"Email: {self.email}, "
                   f"Language: {self.language}, "
                   f"XTTS Supported: {self.is_xtts_supported}")

        if self.embedding is not None:
            if self.embedding.ndim > 1:
                details += f", Embedding first: {self.embedding[0, :5]}"
            else:
                details += f", Embedding: {self.embedding[:5]}"

        if self.gpt_cond_latent is not None:
            if self.gpt_cond_latent.ndim > 1:
                details += f", GPT first: {self.gpt_cond_latent[0, :5]}"
            else:
                details += f", GPT: {self.gpt_cond_latent[:5]}"

        return details
