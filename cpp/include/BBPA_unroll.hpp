#pragma once

#define BBPA_UNROLL_1(partition_id, ex, state, curr_subjs) \
    partition_id |= (1 & (((ex & (offset_to_state(state))) - ex) >> 31));

#define BBPA_UNROLL_2(partition_id, ex, state, curr_subjs) \
    BBPA_UNROLL_1(partition_id, ex, state, curr_subjs)     \
    partition_id |= (2 & (((ex & (offset_to_state(state) >> curr_subjs)) - ex) >> 31));

#define BBPA_UNROLL_3(partition_id, ex, state, curr_subjs) \
    BBPA_UNROLL_2(partition_id, ex, state, curr_subjs)                                        \
    partition_id |= (4 & (((ex & (offset_to_state(state) >> (2 * curr_subjs))) - ex) >> 31));

#define BBPA_UNROLL_4(partition_id, ex, state, curr_subjs) \
    BBPA_UNROLL_3(partition_id, ex, state, curr_subjs)                                        \
    partition_id |= (8 & (((ex & (offset_to_state(state) >> (3 * curr_subjs))) - ex) >> 31));

#define BBPA_UNROLL_5(partition_id, ex, state, curr_subjs) \
    BBPA_UNROLL_4(partition_id, ex, state, curr_subjs)                                        \
    partition_id |= (16 & (((ex & (offset_to_state(state) >> (4 * curr_subjs))) - ex) >> 31));

#define BBPA_UNROLL(times, partition_id, ex, state, curr_subjs) \
    BBPA_UNROLL_##times(partition_id, ex, state, curr_subjs)
