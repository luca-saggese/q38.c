# MTP status

The checkpoint config declares one hybrid MTP layer with full-attention
layer type, shared/dedicated embedding flags as recorded in the M9 spec, and
rope theta 10,000,000. It does not define proposal/verification semantics or
runtime tensor mapping sufficiently for a safe implementation. MTP remains
DISABLED and is not used to judge M8 Q4 quality.
