from typing import List, Optional, Dict

from filelock import FileLock
from openai.api_resources.abstract import engine_api_resource
import openai as turing

from helm.common.cache import Cache, CacheConfig
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from .client import Client, wrap_request_time, truncate_sequence
from .openai_client import ORIGINAL_COMPLETION_ATTRIBUTES
import requests


class MSRAClient(Client):
    """
    Client for the Microsoft's Megatron-Turing NLG models (https://arxiv.org/abs/2201.11990).

    According to the internal documentation: https://github.com/microsoft/turing-academic-TNLG,
    "the model will generate roughly 3 tokens per second. The response will be returned once
    all tokens have been generated."
    """

    @staticmethod
    def convert_to_raw_request(request: Request) -> Dict:
        return {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            # Despite what was stated here: https://github.com/microsoft/turing-academic-TNLG#api-parameters,
            # their API supports at most one stop sequence. Pass in the first one for now and handle the rest
            # of the stop sequences during post processing (see `truncate_sequence` below).
            "stop": None if len(request.stop_sequences) == 0 else request.stop_sequences[0],
            "top_p": request.top_p,
            "echo": request.echo_prompt,
        }

    def __init__(
        self,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
    ):
        # Adapted from their documentation: https://github.com/microsoft/turing-academic-TNLG
        class EngineAPIResource(engine_api_resource.EngineAPIResource):
            @classmethod
            def class_url(
                cls, engine: Optional[str] = None, api_type: Optional[str] = None, api_version: Optional[str] = None
            ) -> str:
                return f"/{engine}/inference"

        self.org_id: Optional[str] = org_id
        self.api_key: Optional[str] = True
        self.api_base: str = "https://10.184.185.28:443/query"
        self.completion_attributes = (EngineAPIResource,) + ORIGINAL_COMPLETION_ATTRIBUTES[1:]

        self.cache = Cache(cache_config)

        # The Microsoft Turing server only allows a single request at a time, so acquire a
        # process-safe lock before making a request.
        # https://github.com/microsoft/turing-academic-TNLG#rate-limitations
        #
        # Since the model will generate roughly three tokens per second and the max context window
        # is 2048 tokens, we expect the maximum time for a request to be fulfilled to be 700 seconds.

    def make_request(self, request: Request) -> RequestResult:
        """
        Make a request for the Microsoft MT-NLG models.

        They mimicked the OpenAI completions API, but not all the parameters are supported.

        Supported parameters:
            engine
            prompt
            temperature
            max_tokens
            best_of
            logprobs
            stop ("Only a single "stop" value (str) is currently supported.")
            top_p
            echo
            n (Not originally supported, but we simulate n by making multiple requests)

        Not supported parameters:
            presence_penalty
            frequency_penalty
        """
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request = MSRAClient.convert_to_raw_request(request)
        completions: List[Sequence] = []
        request_time = 0
        request_datetime: Optional[int] = None
        all_cached = True

        # API currently only supports 1 completion at a time, so we have to hit it multiple times.
        for completion_index in range(request.num_completions):
            try:

                def do_it():
                    url = self.api_base
                    data = {'prompt': raw_request['prompt']}

                    response = requests.post(url, data=data)

                    print(response.status_code) # Print the response status code
                    print(response.json()) # Print the response body as JSON
                    return response

                def fail():
                    raise RuntimeError(
                        f"The result has not been uploaded to the cache for the following request: {cache_key}"
                    )

                # We want to make `request.num_completions` fresh requests,
                # cache key should contain the completion_index.
                cache_key = Client.make_cache_key({"completion_index": completion_index, **raw_request}, request)
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it if self.api_key else fail))
            except turing.error.OpenAIError as e:
                error: str = f"OpenAI (Turing API) error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

            sequence_logprob = 0
            tokens: List[Token] = []

            for text, logprob in zip(
                response["tokens"], response["token_logprobs"]
            ):
                tokens.append(Token(text=text, logprob=logprob or 0, top_logprobs=dict({})))
                sequence_logprob += logprob or 0

            completion = Sequence(
                text=response["text"],
                logprob=response["logprobs"],
                tokens=tokens,
                finish_reason={"reason": response["finish_reason"]},
            )
            completion = truncate_sequence(completion, request)
            completions.append(completion)

            request_time += response["request_time"]
            # Use the datetime from the first completion because that's when the request was fired
            request_datetime = request_datetime or response.get("request_datetime")
            all_cached = all_cached and cached

        return RequestResult(
            success=True,
            cached=all_cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to tokenize.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("Use the HuggingFaceClient to decode.")
