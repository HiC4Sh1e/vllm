from typing import Any, Optional, Callable, Dict
from dataclasses import dataclass, field
from asyncio import AbstractEventLoop

from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.core_client import MPClient

import time
import threading

SECOND_TO_MS = 1000

@dataclass
class BufferedResponse:
    request_id: str
    output: list[Any]
    is_ended: Optional[bool] = False
    have_sent_prefill: Optional[bool] = False
    last_processed_time: Optional[float] = 0.0
    slo_requirement: Optional[dict] = field(default_factory=dict)
    engine_index: Optional[int] = 0
    is_aborted : Optional[bool] = False

class BufferResponseProcessor():
    def __init__(self,
            process_callback: Callable[[Any], Any],
            engine_num: Optional[int] = 1
        ):
        """
        Init BufferResponseProcessor and object is belonged to async_llm
        :param process_callback: func to release responses to request when it meets slo requirements
        :param engine_num: Optional, record the engine num for saving corresponding logs to different loggers in async_llm
        """
        self.process_callback = process_callback
        self.slo_send_factor = 0.95
        self.default_slo_ms = {"TTFT" : 1000, "ITL" : 50}
        self.engine_num = engine_num
        self.response_container : Dict[str, BufferedResponse] = {}

        self._running = True
        self._buffer_response_thread = threading.Thread(
            target=self.process_buffered_response,
            daemon=True
        )
        self._buffer_response_thread.start()

    def add_response(self, response: BufferedResponse) -> None:
        """
        Add BufferedResponse to the BufferResponseProcessor.
        :param response: class BufferedResponse with request_id, output and slo_requirement(optional)
        :return: None
        """
        if response.request_id in self.response_container:
            # update output, engine_index(DP), is_ended for the request in response_container
            self.response_container[response.request_id].output.extend(response.output)
            self.response_container[response.request_id].engine_index = response.engine_index
            self.response_container[response.request_id].is_ended = response.is_ended
        else:
            # add new request to response_container
            self.response_container[response.request_id] = response

    def abort_request(self, request_id: str) -> None:
        """
        Remove the request from response_container once it is aborted
        :param request_id: str
        :return: None
        """
        if request_id in self.response_container:
            self.response_container[request_id].is_aborted = True

    def slo_checker(self) -> list[Any]:
        """
        To filter outputs that are approaching to SLO requirements
        :return: list[Any]
        """
        global SECOND_TO_MS
        to_send_output = list([] for _ in range(self.engine_num))
        to_update_response = list([] for _ in range(self.engine_num))
        processing_responses = list(self.response_container.keys())

        for req_id in processing_responses:
            req_response = self.response_container[req_id]

            if req_response.is_aborted:
                to_update_response[req_response.engine_index].append(req_id)
                continue

            if req_response.is_ended:
                # if the req is finished or ended, send all the responses
                to_send_output[req_response.engine_index].extend(req_response.output)
                to_update_response[req_response.engine_index].append(req_id)
            else:
                target_slo_str = "ITL" if req_response.have_sent_prefill else "TTFT"

                if target_slo_str in req_response.slo_requirement:
                    target_slo = req_response.slo_requirement[target_slo_str]
                else:
                    target_slo = self.default_slo_ms[target_slo_str]

                if ((time.time() - req_response.last_processed_time) * SECOND_TO_MS > self.slo_send_factor * target_slo
                        and len(req_response.output) > 0):
                    to_send_output[req_response.engine_index].append(req_response.output.pop())
                    to_update_response[req_response.engine_index].append(req_id)

        for engine_index in range(self.engine_num):
            for req_id in to_update_response[engine_index]:
                self._update_reponse_container(req_id)

        return to_send_output

    def process_buffered_response(self) -> None:
        """
        Loop to check slo in response_container and release buffered responses
        :return: None
        """
        while self._running:
            to_send_output = self.slo_checker()
            for engine_index in range(self.engine_num):
                if len(to_send_output[engine_index]) > 0:
                    self.process_callback(outputs = to_send_output[engine_index], engine_index = engine_index)
            time.sleep(0.001)

    def _update_reponse_container(self, req_id: str) -> None:
        """
        Update the request's info in response_container
        :param req_id: str, request id
        :return: None
        """
        response = self.response_container[req_id]
        #remove request once it is aborted or ended
        if response.is_ended or response.is_aborted:
            del self.response_container[req_id]
            return

        # update whether send the first token
        if not self.response_container[req_id].have_sent_prefill:
            self.response_container[req_id].have_sent_prefill = True
        self.response_container[req_id].last_processed_time = time.time()

    def stop(self) -> None:
        """
        End buffer_response_thread
        :return: None
        """
        self._running = False
        if self._buffer_response_thread and self._buffer_response_thread.is_alive():
            self._buffer_response_thread.join(timeout=10.0)


def bind_fixed_params(*fixed_args: Any, **fixed_kwargs: Any):
    """
    Decorator to bind fixed parameters or arguments to callback function
    :param fixed_args: fixed positional arguments
    :param fixed_kwargs: fixed named arguments
    :return: callback func with fixed parameters or arguments
    """
    def decorator(callback: Callable[..., None]) -> Callable[..., None]:
        def wrapped(*dynamic_args: Any, **dynamic_kwargs: Any) -> None:
            callback(*fixed_args, *dynamic_args, **fixed_kwargs, **dynamic_kwargs)
        return wrapped
    return decorator


def vllm_process_callback(loop: AbstractEventLoop, engine_core: MPClient, outputs: list[EngineCoreOutput], engine_index : Optional[int] = 0) -> None:
    """
    VLLM callback function used in BufferResponseProcessor
    :param loop: the loop in async_llm to receive buffered responses from BufferResponseProcessor
    :param engine_core: engine_core in async_llm
    :param outputs: buffered responses to release
    :param engine_index: Optional, use the index to release info to specific logger in async_llm
    :return: None
    """
    engine_core_outputs = EngineCoreOutputs(outputs=outputs, engine_index = engine_index, is_buffered_outputs=True)
    loop.call_soon_threadsafe(engine_core.outputs_queue.put_nowait, engine_core_outputs)