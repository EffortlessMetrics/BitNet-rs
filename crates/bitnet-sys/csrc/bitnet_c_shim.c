#include "bitnet_c.h"

// TODO: include real C++ headers and implement; for now these fail cleanly
struct bitnet_model {};
struct bitnet_ctx   {};

bitnet_model_t* bitnet_model_new_from_file(const char* gguf_path) { (void)gguf_path; return 0; }
void            bitnet_model_free(bitnet_model_t* m) { (void)m; }
bitnet_ctx_t*   bitnet_context_new(bitnet_model_t* m, const bitnet_params_t* p) { (void)m; (void)p; return 0; }
void            bitnet_context_free(bitnet_ctx_t* c) { (void)c; }
int             bitnet_tokenize(bitnet_model_t* m, const char* t, int a, int s, int32_t* o, int cap) { (void)m;(void)t;(void)a;(void)s;(void)o;(void)cap; return -1; }
int             bitnet_eval(bitnet_ctx_t* c, const int32_t* ids, int n, float* out, int cap) { (void)c;(void)ids;(void)n;(void)out;(void)cap; return -1; }
int             bitnet_decode_greedy(bitnet_ctx_t* c, int32_t* ids, int maxn, int eos, float temp) { (void)c;(void)ids;(void)maxn;(void)eos;(void)temp; return -1; }
