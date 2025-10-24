from vllm import SamplingParams
from vllm.logger import init_logger
from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.multi_block_pool import MultiBlockPool
from vllm.v1.request import Request

logger = init_logger(__name__)


def test_chunked_local_attention_possible_cached_prefix():
    num_total_blocks = 3
    block_size = 2
    num_pools = 2

    cached_token_ids = [0, 114, 514, 1919, 810]

    global_block_pool = MultiBlockPool(num_total_blocks,
                                       enable_caching=True,
                                       enable_kv_cache_events=True,
                                       num_pools=num_pools)

    init_none_hash(sha256)
    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(ignore_eos=False,
                                     max_tokens=100,
                                     stop_token_ids=[1],
                                     prompt_logprobs=1024)
    cache_request = Request(request_id='cache_request',
                            prompt_token_ids=cached_token_ids,
                            sampling_params=sampling_params,
                            pooling_params=None,
                            eos_token_id=1,
                            block_hasher=block_hasher)

    num_cached_blocks = len(cached_token_ids) // block_size
    assert num_cached_blocks % num_pools == 0
    cached_blocks = global_block_pool.get_new_blocks(num_cached_blocks)
    assert len(cached_blocks) == num_cached_blocks

    global_block_pool.cache_full_blocks(request=cache_request,
                                        blocks=cached_blocks,
                                        num_cached_blocks=0,
                                        num_full_blocks=num_cached_blocks,
                                        block_size=block_size,
                                        kv_cache_group_id=0)
    num_free_blocks = global_block_pool.get_num_free_blocks_by_pool([0])
    assert (num_free_blocks == num_total_blocks -
            (num_cached_blocks // num_pools) - 1)
    usage = global_block_pool.get_usage()
    assert usage > 0
    events = global_block_pool.take_events()
    assert len(events) == 1

    global_block_pool.free_blocks(cached_blocks)

    num_free_blocks = global_block_pool.get_num_free_blocks()
    assert num_free_blocks == (num_total_blocks - 1) * num_pools
    usage = global_block_pool.get_usage()
    assert usage == 0

    cached_block = global_block_pool.get_cached_block(
        block_hash=cache_request.block_hashes[0],
        kv_cache_group_ids=[0])
    assert len(cached_block) == 1

    global_block_pool.touch((cached_block,))
    usage = global_block_pool.get_usage()
    assert usage > 0
    global_block_pool.free_blocks(cached_block)

    num_free_blocks = (num_total_blocks - 1) * num_pools
    global_block_pool.get_new_blocks(num_free_blocks)
    assert len(global_block_pool.cached_block_hash_to_block) == 0
