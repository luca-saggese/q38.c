#ifndef Q38_DIAGNOSTICS_H
#define Q38_DIAGNOSTICS_H

#if defined(Q38_DIAGNOSTICS) && Q38_DIAGNOSTICS
#define Q38_DIAG_ENABLED 1
#define Q38_DIAG_ONLY(statement) do { statement; } while (0)
#define Q38_DIAG_EXPR(expression, fallback) (expression)
#else
#define Q38_DIAG_ENABLED 0
#define Q38_DIAG_ONLY(statement) do { } while (0)
#define Q38_DIAG_EXPR(expression, fallback) (fallback)
#endif

#endif /* Q38_DIAGNOSTICS_H */
