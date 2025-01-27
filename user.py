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
