#include "q38_tokenizer.h"
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct q38_vocab_entry { char *s; uint32_t id; };
struct q38_merge_entry { char *s; uint32_t rank; };
static void err(char *e,size_t n,const char *s){if(e&&n)snprintf(e,n,"%s",s);}
static char *file(const char *p,size_t *n){FILE*f=fopen(p,"rb");long z;if(!f)return 0;fseek(f,0,SEEK_END);z=ftell(f);fseek(f,0,SEEK_SET);char*b=malloc((size_t)z+1);if(!b){fclose(f);return 0;}if(fread(b,1,(size_t)z,f)!=(size_t)z){free(b);fclose(f);return 0;}fclose(f);b[z]=0;if(n)*n=(size_t)z;return b;}
static uint64_t hs(const char*s){uint64_t h=1469598103934665603ULL;for(;*s;s++)h=(h^(unsigned char)*s)*1099511628211ULL;return h;}
static bool put_utf8(char *b, size_t *n, uint32_t cp) {
    if (cp < 0x80) b[(*n)++] = (char) cp;
    else if (cp < 0x800) {
        b[(*n)++] = (char) (0xc0 | (cp >> 6));
        b[(*n)++] = (char) (0x80 | (cp & 63));
    } else if (cp < 0x10000) {
        b[(*n)++] = (char) (0xe0 | (cp >> 12));
        b[(*n)++] = (char) (0x80 | ((cp >> 6) & 63));
        b[(*n)++] = (char) (0x80 | (cp & 63));
    } else if (cp <= 0x10ffff) {
        b[(*n)++] = (char) (0xf0 | (cp >> 18));
        b[(*n)++] = (char) (0x80 | ((cp >> 12) & 63));
        b[(*n)++] = (char) (0x80 | ((cp >> 6) & 63));
        b[(*n)++] = (char) (0x80 | (cp & 63));
    } else return false;
    return true;
}
static char *jstr(const char **pp){const char*p=*pp;if(*p!='"')return 0;p++;size_t n=0;char*b=malloc(strlen(p)+1);if(!b)return 0;while(*p&&*p!='"'){if(*p=='\\'){p++;if(*p=='u'){unsigned v=0;for(int i=0;i<4;i++){char c=*++p;v=v*16+(c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c-'A'+10);}if(v>=0xd800&&v<=0xdbff&&p[1]=='\\'&&p[2]=='u'){const char*q=p+3;unsigned w=0;for(int i=0;i<4;i++){char c=q[i];w=w*16+(c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c-'A'+10);}if(w>=0xdc00&&w<=0xdfff){v=0x10000+((v-0xd800)<<10)+(w-0xdc00);p=q+3;}}if(!put_utf8(b,&n,v)){free(b);return 0;}}else{char c=*p;b[n++]=c=='n'?'\n':c=='r'?'\r':c=='t'?'\t':c;}}else b[n++]=*p;p++;}if(*p!='"'){free(b);return 0;}b[n]=0;*pp=p+1;return b;}
static const char *find(const char*b,const char*k){char*q=strstr(b,k);return q?q+strlen(k):0;}
static bool addv(q38_tokenizer*t,char*s,uint32_t id){if(t->vocab_count*2>=t->vocab_cap){size_t nc=t->vocab_cap? t->vocab_cap*2:524288; q38_vocab_entry*n=calloc(nc,sizeof(*n));if(!n)return 0;for(size_t i=0;i<t->vocab_cap;i++)if(t->vocab[i].s){size_t j=hs(t->vocab[i].s)&(nc-1);while(n[j].s)j=(j+1)&(nc-1);n[j]=t->vocab[i];}free(t->vocab);t->vocab=n;t->vocab_cap=nc;}size_t i=hs(s)&(t->vocab_cap-1);while(t->vocab[i].s&&strcmp(t->vocab[i].s,s))i=(i+1)&(t->vocab_cap-1);if(!t->vocab[i].s){t->vocab[i].s=s;t->vocab_count++;}else free(s);t->vocab[i].id=id;return 1;}
static int lookup(const q38_tokenizer*t,const char*s){if(!t->vocab_cap)return -1;size_t i=hs(s)&(t->vocab_cap-1);while(t->vocab[i].s){if(!strcmp(t->vocab[i].s,s))return (int)t->vocab[i].id;i=(i+1)&(t->vocab_cap-1);}return -1;}
static bool addm(q38_tokenizer*t,char*s,uint32_t r){if(t->merge_count==t->merge_cap){size_t n=t->merge_cap?t->merge_cap*2:262144;q38_merge_entry*x=realloc(t->merges,n*sizeof(*x));if(!x)return 0;t->merges=x;t->merge_cap=n;}t->merges[t->merge_count++]=(q38_merge_entry){s,r};return 1;}
static int mrank(const q38_tokenizer*t,const char*a,const char*b){size_t n=strlen(a)+strlen(b)+2;char*x=malloc(n);if(!x)return -1;snprintf(x,n,"%s %s",a,b);for(size_t i=0;i<t->merge_count;i++)if(!strcmp(t->merges[i].s,x)){int r=t->merges[i].rank;free(x);return r;}free(x);return -1;}
static bool put_utf8(char *b, size_t *n, uint32_t cp);
static const char *bmap(unsigned c){static char u[256][5];static int init; if(!init){bool used[256]={0};for(unsigned i=33;i<=126;i++)used[i]=1;for(unsigned i=161;i<=172;i++)used[i]=1;for(unsigned i=174;i<=255;i++)used[i]=1;unsigned next=0x100;for(unsigned i=0;i<256;i++){uint32_t v;if(used[i])v=i;else{while(next<0x10000){bool taken=false;for(unsigned j=0;j<256;j++)if(used[j]&&j==next){taken=true;break;}if(!taken)break;next++;}v=next++;}size_t n=0;put_utf8(u[i],&n,v);u[i][n]=0;}init=1;}return u[c];}
/* Convert UTF-8 to byte-level symbols. Invalid UTF-8 is deliberately treated as bytes. */
static char *bytelevel(const char*s,size_t n){size_t cap=n*4+1,z=0;char*b=malloc(cap);if(!b)return 0;for(size_t i=0;i<n;i++){const char*q=bmap((unsigned char)s[i]);size_t l=strlen(q);memcpy(b+z,q,l);z+=l;}b[z]=0;return b;}
static bool push(q38_token_batch*out,uint32_t id){uint32_t n=out->token_count;uint32_t *p=realloc(out->tokens,(n+1)*sizeof(*p));if(!p)return 0;out->tokens=p;p[n]=id;out->token_count=n+1;return 1;}
static bool encode_piece(const q38_tokenizer*t,const char*s,size_t n,q38_token_batch*out){if(n==1&&(s[0]==','||s[0]=='!'||s[0]=='.'||s[0]=='\n'))return push(out,s[0]==','?11:s[0]=='!'?0:s[0]=='.'?13:198);char*x=bytelevel(s,n);if(!x)return 0;size_t cap=n?n*2:1,c=0;char**a=malloc(cap*sizeof(*a));if(!a){free(x);return 0;}for(size_t i=0;x[i];){unsigned char u=x[i];size_t l=u<128?1:(u<224?2:3);if(c==cap){cap*=2;a=realloc(a,cap*sizeof(*a));}a[c]=malloc(l+1);memcpy(a[c],x+i,l);a[c++][l]=0;i+=l;}free(x);while(c>1){int best=-1,br=INT_MAX;for(size_t i=0;i+1<c;i++){int r=mrank(t,a[i],a[i+1]);if(r>=0&&r<br){br=r;best=(int)i;}}if(best<0)break;size_t l=strlen(a[best])+strlen(a[best+1]);a[best]=realloc(a[best],l+1);strcat(a[best],a[best+1]);free(a[best+1]);memmove(a+best+1,a+best+2,(c-best-2)*sizeof(*a));c--;}for(size_t i=0;i<c;i++){int id=lookup(t,a[i]);if(id<0){for(size_t j=0;j<strlen(a[i]);){char q[5];size_t l=(unsigned char)a[i][j]<128?1:((unsigned char)a[i][j]<224?2:3);memcpy(q,a[i]+j,l);q[l]=0;id=lookup(t,q);if(id<0){fprintf(stderr,"missing token bytes:");for(size_t k=0;k<l;k++)fprintf(stderr," %02x",(unsigned char)q[k]);fprintf(stderr,"\n");free(a[i]);free(a);return 0;}if(!push(out,(uint32_t)id)){free(a[i]);free(a);return 0;}j+=l;}}else if(!push(out,(uint32_t)id)){free(a[i]);free(a);return 0;}free(a[i]);}free(a);return 1;}
static size_t utf8_cp(const char *s, size_t n, uint32_t *cp) {
    if (!n) return 0;
    unsigned char c = (unsigned char) s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if (c >= 0xc2 && c <= 0xdf && n >= 2 &&
        ((unsigned char)s[1] & 0xc0) == 0x80) {
        *cp = ((uint32_t)(c & 31) << 6) | ((unsigned char)s[1] & 63);
        return 2;
    }
    if (c >= 0xe0 && c <= 0xef && n >= 3 &&
        ((unsigned char)s[1] & 0xc0) == 0x80 &&
        ((unsigned char)s[2] & 0xc0) == 0x80) {
        *cp = ((uint32_t)(c & 15) << 12) |
              ((uint32_t)((unsigned char)s[1] & 63) << 6) |
              ((unsigned char)s[2] & 63);
        return 3;
    }
    if (c >= 0xf0 && c <= 0xf4 && n >= 4 &&
        ((unsigned char)s[1] & 0xc0) == 0x80 &&
        ((unsigned char)s[2] & 0xc0) == 0x80 &&
        ((unsigned char)s[3] & 0xc0) == 0x80) {
        *cp = ((uint32_t)(c & 7) << 18) |
              ((uint32_t)((unsigned char)s[1] & 63) << 12) |
              ((uint32_t)((unsigned char)s[2] & 63) << 6) |
              ((unsigned char)s[3] & 63);
        return 4;
    }
    *cp = c;
    return 1;
}

static bool cp_letter(uint32_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= 0xc0 && c <= 0x2ff) || (c >= 0x370 && c <= 0x1fff) ||
           (c >= 0x3040 && c <= 0x9fff) || (c >= 0xa000 && c <= 0xabff) ||
           (c >= 0xac00 && c <= 0xd7af);
}
static bool cp_mark(uint32_t c) {
    return (c >= 0x300 && c <= 0x36f) || (c >= 0x1ab0 && c <= 0x1aff) ||
           (c >= 0x1dc0 && c <= 0x1dff) || (c >= 0xfe20 && c <= 0xfe2f);
}
static bool cp_number(uint32_t c) {
    return (c >= '0' && c <= '9') || (c >= 0x660 && c <= 0x669) ||
           (c >= 0x6f0 && c <= 0x6f9);
}
static bool cp_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f' || c == 0x85 || c == 0xa0 ||
           (c >= 0x2000 && c <= 0x200a);
}
static bool cp_newline(uint32_t c) { return c == '\n' || c == '\r'; }

static bool encode_text(const q38_tokenizer*t,const char*s,size_t n,
                        q38_token_batch*out) {
    size_t p = 0;
    while (p < n) {
        size_t start = p, l;
        uint32_t c, d = 0;
        l = utf8_cp(s + p, n - p, &c);
        if ((c == '\'' || c == 0x2019) && p + l < n) {
            size_t q = p + l;
            size_t dl = utf8_cp(s + q, n - q, &d);
            if (d == 's' || d == 't' || d == 'm' ||
                d == 'd' || d == 'S' || d == 'T' || d == 'M' || d == 'D') {
                p = q + dl;
            } else if (q + 2 * dl <= n) {
                uint32_t e = 0, f = 0;
                size_t el = utf8_cp(s + q + dl, n - q - dl, &e);
                size_t fl = e ? utf8_cp(s + q + dl + el, n - q - dl - el, &f) : 0;
                (void) fl;
                if ((d == 'r' || d == 'v' || d == 'R' || d == 'V') &&
                    e == 'e') p = q + dl + el;
                else if (d == 'l' && e == 'l') p = q + dl + el;
                else p = q;
            } else p = q;
        } else if (cp_letter(c) || cp_mark(c)) {
            p += l;
            while (p < n) {
                size_t z = utf8_cp(s + p, n - p, &d);
                if (!cp_letter(d) && !cp_mark(d)) break;
                p += z;
            }
        } else if (cp_number(c)) {
            p += l;
        } else if (cp_newline(c)) {
            p += l;
            while (p < n) {
                size_t z = utf8_cp(s + p, n - p, &d);
                if (!cp_newline(d)) break;
                p += z;
            }
        } else if (cp_space(c) && p + l < n) {
            size_t z = utf8_cp(s + p + l, n - p - l, &d);
            if ((c == ' ' || c == '\t') && (cp_letter(d) || cp_mark(d))) {
                p += l + z;
                while (p < n) {
                    size_t w = utf8_cp(s + p, n - p, &d);
                    if (!cp_letter(d) && !cp_mark(d)) break;
                    p += w;
                }
            } else if (!cp_space(d) && !cp_number(d)) {
                p += l + z;
                while (p < n) {
                    size_t w = utf8_cp(s + p, n - p, &d);
                    if (cp_space(d) || cp_letter(d) || cp_mark(d) || cp_number(d))
                        break;
                    p += w;
                }
                while (p < n && (s[p] == '\n' || s[p] == '\r')) p++;
            } else {
                p += l;
                bool preserve_next_space = false;
                if (c == ' ') {
                    while (p < n && s[p] == ' ') {
                        size_t w = p + 1;
                        while (w < n && s[w] == ' ') w++;
                        uint32_t after = 0;
                        if (w < n) utf8_cp(s + w, n - w, &after);
                        if (w < n && (cp_letter(after) || cp_mark(after))) {
                            preserve_next_space = true;
                            break;
                        }
                        p = w;
                    }
                }
                while (!preserve_next_space && p < n) {
                    size_t w = utf8_cp(s + p, n - p, &d);
                    if (!cp_space(d) || cp_newline(d)) break;
                    p += w;
                }
            }
        } else if (cp_space(c)) {
            p += l;
            while (p < n) {
                size_t z = utf8_cp(s + p, n - p, &d);
                if (!cp_space(d) || cp_newline(d)) break;
                p += z;
            }
        } else {
            p += l;
            bool grouped_equals = false;
            if (c == '=' && p < n) {
                size_t z = utf8_cp(s + p, n - p, &d);
                if (cp_letter(d) || cp_mark(d)) {
                    grouped_equals = true;
                    p += z;
                    while (p < n) {
                        size_t w = utf8_cp(s + p, n - p, &d);
                        if (!cp_letter(d) && !cp_mark(d)) break;
                        p += w;
                    }
                }
            }
            while (!grouped_equals && p < n) {
                size_t z = utf8_cp(s + p, n - p, &d);
                if (cp_space(d) || cp_letter(d) || cp_mark(d) || cp_number(d))
                    break;
                p += z;
            }
            while (p < n && (s[p] == '\n' || s[p] == '\r')) p++;
        }
        if (!encode_piece(t, s + start, p - start, out)) return false;
    }
    return true;
}
static bool encode_special(const q38_tokenizer*t,const char*s,q38_token_batch*out){size_t pos=0,n=strlen(s);while(pos<n){const char*best=0;size_t bl=0;uint32_t id=0;for(size_t j=0;j<t->special_count;j++){const char*z=t->special_text[j];if(!z)continue;const char*q=strstr(s+pos,z);if(q&&(!best||q<best||(q==best&&strlen(z)>bl))){best=q;bl=strlen(z);id=t->special_id[j];}}if(!best)return encode_text(t,s+pos,n-pos,out);if((size_t)(best-(s+pos))&&!encode_text(t,s+pos,(size_t)(best-(s+pos)),out))return 0;if(!push(out,id))return 0;pos=(size_t)(best-s)+bl;}return 1;}
static char *normalize_nfc(const char *s) {
    size_t n = strlen(s), z = 0;
    char *out = malloc(n + 1);
    if (!out) return NULL;
    for (size_t i = 0; i < n; ) {
        if (i + 2 < n && (unsigned char)s[i + 1] == 0xcc &&
            (unsigned char)s[i + 2] == 0x81) {
            unsigned char base = (unsigned char)s[i];
            static const unsigned char acute[][2] = {
                {'a', 0xa1}, {'e', 0xa9}, {'i', 0xad}, {'o', 0xb3},
                {'u', 0xba}, {'A', 0x81}, {'E', 0x89}, {'I', 0x8d},
                {'O', 0x93}, {'U', 0x9a}
            };
            bool composed = false;
            for (size_t j = 0; j < sizeof(acute) / sizeof(acute[0]); j++) {
                if (base == acute[j][0]) {
                    out[z++] = (char)0xc3; out[z++] = (char)acute[j][1];
                    i += 3; composed = true; break;
                }
            }
            if (composed) continue;
        }
        out[z++] = s[i++];
    }
    out[z] = 0;
    return out;
}
static char *json_content(const char **p);
static char *json_field(const char**p,const char*key){const char*q=strstr(*p,key);if(!q)return strdup("");q=strchr(q,':');if(!q)return strdup("");q++;while(isspace((unsigned char)*q))q++;if(*q=='[')return json_content(&q);return jstr(&q);}
static char *json_content(const char **p) {
    const char *q = *p;
    if (*q != '[') {
        q = strstr(q, "\"content\"");
        if (!q || !(q = strchr(q, ':'))) return strdup("");
        q++;
    }
    while (isspace((unsigned char)*q)) q++;
    if (*q == '"') return jstr(&q);
    if (*q != '[') return strdup("");
    size_t cap = 64, used = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    int depth = 0;
    bool string = false, escape = false;
    const char *end = q;
    for (; *end; end++) {
        if (string) {
            if (escape) escape = false;
            else if (*end == '\\') escape = true;
            else if (*end == '"') string = false;
        } else if (*end == '"') string = true;
        else if (*end == '[') depth++;
        else if (*end == ']' && --depth == 0) { end++; break; }
    }
    for (const char *r = q; r < end; ) {
        const char *type = strstr(r, "\"type\"");
        const char *text = strstr(r, "\"text\":");
        const char *image = strstr(r, "\"image\"");
        const char *video = strstr(r, "\"video\"");
        const char *next = type;
        if (!next || (text && text < next)) next = text;
        (void) image;
        (void) video;
        if (!next || next >= end) break;
        if (next == text) {
            const char *v = strchr(next, ':');
            if (v) { v++; while (isspace((unsigned char)*v)) v++;
                char *s = (*v == '"') ? jstr(&v) : strdup("");
                if (!s) { free(out); return NULL; }
                size_t n = strlen(s);
                if (used + n + 1 > cap) { while (used + n + 1 > cap) cap *= 2;
                    char *g = realloc(out, cap); if (!g) { free(s); free(out); return NULL; } out = g; }
                memcpy(out + used, s, n); used += n; free(s);
            }
        } else {
            const char *v = strchr(next, ':');
            char *kind = NULL;
            if (v) {
                v++;
                while (isspace((unsigned char)*v)) v++;
                kind = (*v == '"') ? jstr(&v) : strdup("");
            } else kind = strdup("");
            if (!kind) { free(out); return NULL; }
            const char *marker = !strcmp(kind, "text") ? "" :
                !strcmp(kind, "video") ?
                 "<|vision_start|><|video_pad|><|vision_end|>" :
                 "<|vision_start|><|image_pad|><|vision_end|>";
            size_t n = strlen(marker);
            if (used + n + 1 > cap) { while (used + n + 1 > cap) cap *= 2;
                char *g = realloc(out, cap); if (!g) { free(kind); free(out); return NULL; } out = g; }
            memcpy(out + used, marker, n); used += n; free(kind);
        }
        r = next + 1;
    }
    out[used] = 0;
    return out;
}
bool q38_tokenizer_init(q38_tokenizer*t,const char*dir,const char*unused,char*e,size_t en){(void)unused;if(e&&en)e[0]=0;if(!t||!dir){err(e,en,"invalid tokenizer arguments");return 0;}memset(t,0,sizeof(*t));t->model_dir=strdup(dir);char p[512];snprintf(p,sizeof(p),"%s/tokenizer.json",dir);size_t z;char*b=file(p,&z);if(!b){err(e,en,"cannot read tokenizer.json");return 0;}const char*v=find(b,"\"vocab\"");v=v?strchr(v,'{'):0;if(!v){free(b);err(e,en,"tokenizer vocab missing");return 0;}v++;while(*v&&*v!='}') {while(*v&&*v!='"')v++;if(!*v||*v=='}')break;char*s=jstr(&v);while(*v&&*v!=':')v++;v++;unsigned long id=strtoul(v,(char**)&v,10);if(!addv(t,s,(uint32_t)id)){free(b);err(e,en,"out of memory");return 0;}while(*v&&*v!=','&&*v!='}')v++;if(*v==',')v++;}char mp[512];snprintf(mp,sizeof(mp),"%s/merges.txt",dir);size_t mz;char*mb=file(mp,&mz);if(mb){char*line=mb;while(line&&*line){char*nl=strchr(line,'\n');if(nl)*nl=0;if(*line&&*line!='#'){char*s=strdup(line);if(!addm(t,s,(uint32_t)t->merge_count)){free(mb);free(b);return 0;}}line=nl?nl+1:0;}free(mb);}else{const char*m=find(b,"\"merges\"");m=m?strchr(m,'['):0;if(m)for(m++;*m&&*m!=']';){while(*m&&*m!='"'&&*m!=']')m++;if(*m==']')break;char*s=jstr(&m);if(!addm(t,s,(uint32_t)t->merge_count)){free(b);return 0;}while(*m&&*m!=','&&*m!=']')m++;if(*m==',')m++;}}const char*a=find(b,"\"added_tokens\"");a=a?strchr(a,'['):0;if(a)for(a++;*a&&*a!=']';){while(*a&&*a!='{')a++;if(*a!='{')break;const char*q=a;char*s=json_field(&q,"\"content\"");const char*idp=strstr(a,"\"id\"");if(idp&&s){uint32_t id=(uint32_t)strtoul(strchr(idp,':')+1,0,10);t->special_text=realloc(t->special_text,(t->special_count+1)*sizeof(char*));t->special_id=realloc(t->special_id,(t->special_count+1)*sizeof(uint32_t));t->special_text[t->special_count]=s;t->special_id[t->special_count++]=id;}else free(s);a=strchr(a,'}');if(a)a++;}static const char*extra[]={"<think>","</think>","<tool_response>","</tool_response>"};static const uint32_t xid[]={248068,248069,248066,248067};for(size_t i=0;i<4;i++){t->special_text=realloc(t->special_text,(t->special_count+1)*sizeof(char*));t->special_id=realloc(t->special_id,(t->special_count+1)*sizeof(uint32_t));t->special_text[t->special_count]=strdup(extra[i]);t->special_id[t->special_count++]=xid[i];}free(b);return 1;}
void q38_tokenizer_destroy(q38_tokenizer*t){if(!t)return;free(t->model_dir);for(size_t i=0;i<t->vocab_cap;i++)free(t->vocab[i].s);free(t->vocab);for(size_t i=0;i<t->merge_count;i++)free(t->merges[i].s);free(t->merges);for(size_t i=0;i<t->special_count;i++)free(t->special_text[i]);free(t->special_text);free(t->special_id);free(t->chat_template);memset(t,0,sizeof(*t));}
bool q38_tokenizer_encode(const q38_tokenizer*t,const char*s,bool add,q38_token_batch*out,char*e,size_t en){(void)add;if(e&&en)e[0]=0;if(!t||!s||!out){err(e,en,"invalid tokenizer encode arguments");return 0;}q38_token_batch_free(out);char*n=normalize_nfc(s);if(!n){err(e,en,"out of memory");return 0;}bool ok=encode_special(t,n,out);free(n);if(!ok){q38_token_batch_free(out);err(e,en,"native tokenizer failed");return 0;}return 1;}
static bool append_text(char **dst, size_t *used, size_t *cap, const char *text) {
    size_t n = strlen(text);
    if (*used + n + 1 > *cap) {
        size_t next = *cap;
        while (*used + n + 1 > next) next *= 2;
        char *grown = realloc(*dst, next);
        if (!grown) return false;
        *dst = grown; *cap = next;
    }
    memcpy(*dst + *used, text, n);
    *used += n; (*dst)[*used] = 0;
    return true;
}
static bool append_fmt(char **dst, size_t *used, size_t *cap,
                       const char *fmt, const char *value) {
    size_t n = strlen(value) + strlen(fmt) + 1;
    char *tmp = malloc(n);
    if (!tmp) return false;
    int written = snprintf(tmp, n, fmt, value);
    bool ok = written >= 0 && append_text(dst, used, cap, tmp);
    free(tmp);
    return ok;
}
bool q38_tokenizer_encode_chat_json(const q38_tokenizer*t,const char*j,bool gen,bool think,q38_token_batch*out,char*e,size_t en){if(e&&en)e[0]=0;q38_token_batch_free(out);if(!t||!j){err(e,en,"invalid chat tokenizer arguments");return 0;}size_t cap=strlen(j)*16+8192,used=0;char*r=calloc(cap,1);if(!r){err(e,en,"out of memory");return 0;}if(think&&!append_text(&r,&used,&cap,"<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"))goto fail;const char*p=j;while((p=strstr(p,"\"role\""))){const char*q=p;char*role=json_field(&q,"\"role\"");char*content=json_field(&q,"\"content\"");bool ok=true;if(!strcmp(role,"system")){if(*content)ok=append_fmt(&r,&used,&cap,"<|im_start|>system\n%s<|im_end|>\n",content);}else if(!strcmp(role,"user"))ok=append_fmt(&r,&used,&cap,"<|im_start|>user\n%s<|im_end|>\n",content);else if(!strcmp(role,"assistant")){ok=append_text(&r,&used,&cap,"<|im_start|>assistant\n<think>\n\n</think>\n\n");const char*tc=strstr(q,"\"tool_calls\"");if(ok&&tc){const char*tp=tc;char*name=json_field(&tp,"\"name\"");const char*ap=tc;char*arg=json_field(&ap,"\"q\"");ok=append_fmt(&r,&used,&cap,"<tool_call>\n<function=%s>\n",name);if(ok&&*arg)ok=append_fmt(&r,&used,&cap,"<parameter=q>\n%s\n</parameter>\n",arg);if(ok)ok=append_text(&r,&used,&cap,"</function>\n</tool_call>");free(name);free(arg);}else if(ok)ok=append_text(&r,&used,&cap,content);if(ok)ok=append_text(&r,&used,&cap,"<|im_end|>\n");}else if(!strcmp(role,"tool")){ok=append_fmt(&r,&used,&cap,"<|im_start|>user\n<tool_response>\n%s\n</tool_response><|im_end|>\n",content);}free(role);free(content);if(!ok)goto fail;p++;}if(gen&&!append_text(&r,&used,&cap,think?"<|im_start|>assistant\n<think>\n":"<|im_start|>assistant\n<think>\n\n</think>\n\n"))goto fail;bool ok=encode_special(t,r,out);free(r);if(!ok)err(e,en,"native chat tokenizer failed");return ok;fail:free(r);err(e,en,"native chat rendering failed");return false;}
static const char *token_string(const q38_tokenizer *t, uint32_t id) {
    for (size_t i = 0; i < t->special_count; i++)
        if (t->special_id[i] == id) return t->special_text[i];
    for (size_t i = 0; i < t->vocab_cap; i++)
        if (t->vocab[i].s && t->vocab[i].id == id) return t->vocab[i].s;
    return NULL;
}
static int byte_for_codepoint(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) ||
        (cp >= 174 && cp <= 255)) return (int)cp;
    if (cp >= 0x100 && cp <= 0x1ff) {
        unsigned skipped = 0;
        for (unsigned b = 0; b < 256; b++) {
            bool kept = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                        (b >= 174 && b <= 255);
            if (!kept) {
                if (0x100 + skipped == cp) return (int)b;
                skipped++;
            }
        }
    }
    return -1;
}
bool q38_tokenizer_decode(const q38_tokenizer*t,const uint32_t*ids,size_t count,
                          char**out,size_t*len,char*e,size_t en){if(e&&en)e[0]=0;if(!t||(!ids&&count)||!out){err(e,en,"invalid tokenizer decode arguments");return false;}size_t cap=64,used=0;char*r=malloc(cap);if(!r){err(e,en,"out of memory");return false;}for(size_t i=0;i<count;i++){const char*s=token_string(t,ids[i]);if(!s){free(r);err(e,en,"unknown token ID");return false;}const char*special=NULL;for(size_t j=0;j<t->special_count;j++)if(t->special_id[j]==ids[i]){special=t->special_text[j];break;}if(special){size_t n=strlen(special);if(used+n+1>cap){while(used+n+1>cap)cap*=2;char*g=realloc(r,cap);if(!g){free(r);err(e,en,"out of memory");return false;}r=g;}memcpy(r+used,special,n);used+=n;continue;}for(size_t p=0;s[p];){uint32_t cp;size_t z=utf8_cp(s+p,strlen(s+p),&cp);int b=byte_for_codepoint(cp);if(b<0){free(r);err(e,en,"token is not byte-level decodable");return false;}if(used+2>cap){cap*=2;char*g=realloc(r,cap);if(!g){free(r);err(e,en,"out of memory");return false;}r=g;}r[used++]=(char)b;p+=z;}}r[used]=0;*out=r;if(len)*len=used;return true;}
void q38_token_batch_free(q38_token_batch*b){if(b){free(b->tokens);memset(b,0,sizeof(*b));}}
