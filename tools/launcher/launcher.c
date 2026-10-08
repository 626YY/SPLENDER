/*
 * SPLENDER.exe：SPLENDER 的启动器。
 *
 * 双击启动 SPLENDER，不出黑窗。步骤：
 *   1. 以自身所在目录为程序目录。
 *   2. 找 64 位 Python 3.11，候选按顺序：
 *        环境变量 SPLENDER_PYTHON_HOME 指定的目录（设了就只用它）
 *        程序目录\runtime\python（以后自带运行环境时用）
 *        注册表 HKCU、HKLM 的 Software\Python\PythonCore\3.11\InstallPath
 *        PATH 里同时有 pythonw.exe 和 python311.dll 的目录
 *   3. 首选内嵌：设好 PYTHONHOME、PYTHONPATH 和 DLL 搜索路径，加载 python311.dll，
 *      调用 Py_Main(exe, -P, -m, splender, 用户参数…)。这样进程就是 SPLENDER.exe 自己，
 *      任务栏和任务管理器里显示的是它。返回 Py_Main 的返回值。
 *      -P 让 Python 不把当前目录放到模块搜索路径最前面，免得工程文件旁边同名的 .py
 *      顶替标准库或 splender 包。
 *   4. 内嵌不成（没有 dll、没有 Py_Main、不是完整的安装）时退回：启动
 *      pythonw.exe -P -m splender …，工作目录为程序目录，等它结束并返回它的退出码。
 *   5. 都不成时弹中文消息框说明情况，退出码非零。
 *
 * 可调（环境变量）：
 *   SPLENDER_PYTHON_HOME   指定 Python 安装目录（放 python311.dll 的目录）
 *   SPLENDER_LAUNCH_MODE   auto（默认）、embed（只用内嵌）、process（只用 pythonw 子进程）
 *   SPLENDER_NO_DIALOG     设为 1 时出错不弹消息框，只写标准错误（给自动化用）
 *
 * 传给 Python 的环境变量：
 *   PYTHONHOME         所用 Python 的安装目录
 *   PYTHONPATH         程序目录（覆盖外部设置，不让别处的库混进来）
 *   SPLENDER_EXE       SPLENDER.exe 的完整路径
 *   SPLENDER_LAUNCHER  embedded 或 process，表示用哪种方式启动
 *
 * 退出码：Python 的退出码；找不到 Python 为 9009；启动 Python 失败为 9010。
 *
 * 运行库用「静态 vcruntime + 系统 UCRT」：不依赖 vcruntime140.dll；又和 python311.dll
 * 共用同一份 ucrtbase.dll，用 _wputenv_s 设的环境变量 Python 一定读得到，而且不经过
 * 936 代码页转换（中文以外的字符也不会丢）。
 */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#define PYTHON_DLL       L"python311.dll"
#define PYTHON_ZIP       L"python311.zip"
#define PYTHONW_EXE      L"pythonw.exe"
#define PYTHON_REG_KEY   L"Software\\Python\\PythonCore\\3.11\\InstallPath"
#define BUNDLED_RUNTIME  L"runtime\\python"
#define PACKAGE_NAME     L"splender"
#define APP_USER_MODEL   L"Splender.App"
#define APP_TITLE        L"SPLENDER"

#define EXIT_NO_PYTHON      9009
#define EXIT_START_FAILED   9010
#define EXIT_OUT_OF_MEMORY  9011

#define MAX_CANDIDATES 32

typedef int(__cdecl *PyMainFn)(int argc, wchar_t **argv);

/* ------------------------------------------------------------------ 内存与字符串 */

static void *xmalloc(size_t size)
{
    void *p = malloc(size ? size : 1);
    if (!p)
        ExitProcess(EXIT_OUT_OF_MEMORY);
    return p;
}

static wchar_t *wdup(const wchar_t *s)
{
    size_t n = wcslen(s) + 1;
    wchar_t *d = (wchar_t *)xmalloc(n * sizeof(wchar_t));
    memcpy(d, s, n * sizeof(wchar_t));
    return d;
}

/* 可增长的宽字符缓冲 */
typedef struct {
    wchar_t *data;
    size_t len;
    size_t cap;
} WBuf;

static void wb_reserve(WBuf *b, size_t extra)
{
    size_t need = b->len + extra + 1;
    size_t cap;
    wchar_t *p;
    if (need <= b->cap)
        return;
    cap = b->cap ? b->cap : 64;
    while (cap < need)
        cap *= 2;
    p = (wchar_t *)realloc(b->data, cap * sizeof(wchar_t));
    if (!p)
        ExitProcess(EXIT_OUT_OF_MEMORY);
    b->data = p;
    b->cap = cap;
}

static void wb_putc(WBuf *b, wchar_t c)
{
    wb_reserve(b, 1);
    b->data[b->len++] = c;
    b->data[b->len] = 0;
}

static void wb_puts(WBuf *b, const wchar_t *s)
{
    size_t n = wcslen(s);
    wb_reserve(b, n);
    memcpy(b->data + b->len, s, n * sizeof(wchar_t));
    b->len += n;
    b->data[b->len] = 0;
}

static void wb_repeat(WBuf *b, wchar_t c, size_t n)
{
    size_t i;
    wb_reserve(b, n);
    for (i = 0; i < n; ++i)
        b->data[b->len++] = c;
    b->data[b->len] = 0;
}

static wchar_t *wb_take(WBuf *b)
{
    wchar_t *p;
    if (!b->data) {
        wb_reserve(b, 0);
        b->data[0] = 0;
    }
    p = b->data;
    b->data = NULL;
    b->len = b->cap = 0;
    return p;
}

static int contains_ci(const wchar_t *hay, const wchar_t *needle)
{
    size_t n = wcslen(needle);
    for (; *hay; ++hay) {
        if (_wcsnicmp(hay, needle, n) == 0)
            return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ 路径 */

static int is_sep(wchar_t c)
{
    return c == L'\\' || c == L'/';
}

/* 去掉末尾的分隔符，但保留盘符根目录的「C:\」 */
static void strip_trailing_separators(wchar_t *path)
{
    size_t n = wcslen(path);
    while (n > 0 && is_sep(path[n - 1])) {
        if (n == 3 && path[1] == L':')
            break;
        path[--n] = 0;
    }
}

static wchar_t *path_join(const wchar_t *dir, const wchar_t *name)
{
    WBuf b = {0};
    wb_puts(&b, dir);
    if (b.len && !is_sep(b.data[b.len - 1]))
        wb_putc(&b, L'\\');
    wb_puts(&b, name);
    return wb_take(&b);
}

static wchar_t *parent_dir(const wchar_t *path)
{
    wchar_t *d = wdup(path);
    wchar_t *cut = NULL;
    wchar_t *p;
    for (p = d; *p; ++p) {
        if (is_sep(*p))
            cut = p;
    }
    if (cut) {
        if (cut == d + 2 && d[1] == L':')
            cut[1] = 0; /* 「C:\SPLENDER.exe」的目录是「C:\」 */
        else
            *cut = 0;
    }
    return d;
}

static int file_exists(const wchar_t *path)
{
    DWORD attr = GetFileAttributesW(path);
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

static int file_in_dir(const wchar_t *dir, const wchar_t *name)
{
    wchar_t *p = path_join(dir, name);
    int ok = file_exists(p);
    free(p);
    return ok;
}

/* 本程序的完整路径 */
static wchar_t *module_path(void)
{
    DWORD cap = MAX_PATH;
    for (;;) {
        wchar_t *buf = (wchar_t *)xmalloc(cap * sizeof(wchar_t));
        DWORD n = GetModuleFileNameW(NULL, buf, cap);
        if (n == 0) {
            free(buf);
            return NULL;
        }
        if (n < cap)
            return buf;
        free(buf);
        if (cap >= 65536)
            return NULL;
        cap *= 2;
    }
}

/* ------------------------------------------------------------------ 环境变量与注册表 */

static wchar_t *env_get(const wchar_t *name)
{
    DWORD n = GetEnvironmentVariableW(name, NULL, 0);
    wchar_t *buf;
    DWORD m;
    if (n == 0)
        return NULL;
    buf = (wchar_t *)xmalloc((size_t)n * sizeof(wchar_t));
    m = GetEnvironmentVariableW(name, buf, n);
    if (m == 0 || m >= n) {
        free(buf);
        return NULL;
    }
    return buf;
}

static int env_flag(const wchar_t *name)
{
    wchar_t *v = env_get(name);
    int on = v && v[0] && wcscmp(v, L"0") != 0;
    free(v);
    return on;
}

/* 同时写进 C 运行库的环境表和系统环境块。Python 读前者，子进程继承后者。 */
static void env_set(const wchar_t *name, const wchar_t *value)
{
    if (_wputenv_s(name, value) != 0)
        SetEnvironmentVariableW(name, value);
}

static wchar_t *expand_env(const wchar_t *s)
{
    DWORD n = ExpandEnvironmentStringsW(s, NULL, 0);
    wchar_t *buf;
    if (n == 0)
        return wdup(s);
    buf = (wchar_t *)xmalloc((size_t)n * sizeof(wchar_t));
    if (ExpandEnvironmentStringsW(s, buf, n) == 0) {
        free(buf);
        return wdup(s);
    }
    return buf;
}

/* 读注册表字符串值；value 为 NULL 表示默认值。没有或为空时返回 NULL。 */
static wchar_t *reg_string(HKEY root, const wchar_t *subkey, const wchar_t *value)
{
    const DWORD flags = RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND;
    DWORD type = 0;
    DWORD size = 0;
    wchar_t *buf;
    if (RegGetValueW(root, subkey, value, flags, &type, NULL, &size) != ERROR_SUCCESS || size == 0)
        return NULL;
    buf = (wchar_t *)xmalloc(size + 2 * sizeof(wchar_t));
    if (RegGetValueW(root, subkey, value, flags, &type, buf, &size) != ERROR_SUCCESS) {
        free(buf);
        return NULL;
    }
    buf[size / sizeof(wchar_t)] = 0;
    if (type == REG_EXPAND_SZ) {
        wchar_t *expanded = expand_env(buf);
        free(buf);
        buf = expanded;
    }
    if (!buf[0]) {
        free(buf);
        return NULL;
    }
    return buf;
}

/* ------------------------------------------------------------------ 找 Python */

typedef struct {
    wchar_t *home;    /* 放 python311.dll 的目录，可能为 NULL */
    wchar_t *pythonw; /* pythonw.exe 的完整路径，可能为 NULL */
} Candidate;

typedef struct {
    Candidate item[MAX_CANDIDATES];
    int count;
} Candidates;

static void add_candidate(Candidates *list, const wchar_t *home, const wchar_t *pythonw)
{
    wchar_t *h = NULL;
    wchar_t *w = NULL;
    int i;
    if (home && home[0]) {
        h = wdup(home);
        strip_trailing_separators(h);
    }
    if (pythonw && pythonw[0] && file_exists(pythonw))
        w = wdup(pythonw);
    if (h && !w) {
        wchar_t *guess = path_join(h, PYTHONW_EXE);
        if (file_exists(guess))
            w = guess;
        else
            free(guess);
    }
    if (!h && !w)
        return;
    for (i = 0; i < list->count; ++i) {
        const Candidate *c = &list->item[i];
        if ((h && c->home && _wcsicmp(c->home, h) == 0) || (!h && w && c->pythonw && _wcsicmp(c->pythonw, w) == 0)) {
            free(h);
            free(w);
            return;
        }
    }
    if (list->count >= MAX_CANDIDATES) {
        free(h);
        free(w);
        return;
    }
    list->item[list->count].home = h;
    list->item[list->count].pythonw = w;
    list->count++;
}

static void collect_candidates(Candidates *list, const wchar_t *prog_dir, const wchar_t *override_home)
{
    static const HKEY roots[2] = {HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
    wchar_t *bundled;
    wchar_t *path;
    int r;

    if (override_home) {
        add_candidate(list, override_home, NULL);
        return;
    }

    bundled = path_join(prog_dir, BUNDLED_RUNTIME);
    if (file_in_dir(bundled, PYTHON_DLL))
        add_candidate(list, bundled, NULL);
    free(bundled);

    for (r = 0; r < 2; ++r) {
        wchar_t *home = reg_string(roots[r], PYTHON_REG_KEY, NULL);
        wchar_t *windowed = reg_string(roots[r], PYTHON_REG_KEY, L"WindowedExecutablePath");
        if (home || windowed)
            add_candidate(list, home, windowed);
        free(home);
        free(windowed);
    }

    path = env_get(L"PATH");
    if (path) {
        wchar_t *ctx = NULL;
        wchar_t *tok;
        for (tok = wcstok_s(path, L";", &ctx); tok; tok = wcstok_s(NULL, L";", &ctx)) {
            size_t n;
            while (*tok == L' ' || *tok == L'"')
                ++tok;
            n = wcslen(tok);
            while (n > 0 && (tok[n - 1] == L' ' || tok[n - 1] == L'"'))
                tok[--n] = 0;
            if (n == 0)
                continue;
            if (file_in_dir(tok, PYTHONW_EXE) && file_in_dir(tok, PYTHON_DLL)) {
                wchar_t *w = path_join(tok, PYTHONW_EXE);
                add_candidate(list, tok, w);
                free(w);
            }
        }
        free(path);
    }
}

/* 是一套完整的安装：有 python311.dll 和标准库。 */
static int has_runtime(const wchar_t *home)
{
    if (!file_in_dir(home, PYTHON_DLL))
        return 0;
    return file_in_dir(home, L"Lib\\os.py") || file_in_dir(home, PYTHON_ZIP);
}

/* 能内嵌：完整安装，且不在微软商店的应用目录里（商店版的库装在虚拟化目录，
   只有通过它自己的 pythonw 启动才找得到）。 */
static int can_embed(const Candidate *c)
{
    return c->home && !contains_ci(c->home, L"\\WindowsApps\\") && has_runtime(c->home);
}

/* ------------------------------------------------------------------ 组参数 */

/* multiprocessing 在 Windows 上用 sys.executable 起子进程，参数形如
   [解释器选项…] -c "from multiprocessing.spawn import spawn_main; …" --multiprocessing-fork。
   内嵌时 sys.executable 就是 SPLENDER.exe，这种调用要原样交给 Python，不能加 -m splender，
   否则每个子进程都会再打开一个主程序。 */
static int is_interpreter_call(int argc, wchar_t **argv)
{
    return argc > 0 && wcscmp(argv[argc - 1], L"--multiprocessing-fork") == 0;
}

/* 交给 Python 的完整参数：[exe, -P, -m, splender, 用户参数…]，以 NULL 结尾。 */
static wchar_t **python_argv(const wchar_t *exe, int argc, wchar_t **argv, int *out_argc)
{
    int interp = is_interpreter_call(argc, argv);
    int n = 1 + (interp ? 0 : 3) + argc;
    wchar_t **v = (wchar_t **)xmalloc(((size_t)n + 1) * sizeof(wchar_t *));
    int k = 0;
    int i;
    v[k++] = wdup(exe);
    if (!interp) {
        v[k++] = wdup(L"-P");
        v[k++] = wdup(L"-m");
        v[k++] = wdup(PACKAGE_NAME);
    }
    for (i = 0; i < argc; ++i)
        v[k++] = wdup(argv[i]);
    v[k] = NULL;
    *out_argc = k;
    return v;
}

/* 按 C 运行库的解析规则给一个参数加引号（反斜杠只在紧挨引号时需要转义）。 */
static void append_quoted(WBuf *b, const wchar_t *arg)
{
    const wchar_t *p;
    if (arg[0] && !wcspbrk(arg, L" \t\n\v\"")) {
        wb_puts(b, arg);
        return;
    }
    wb_putc(b, L'"');
    for (p = arg;; ++p) {
        size_t slashes = 0;
        while (*p == L'\\') {
            ++p;
            ++slashes;
        }
        if (!*p) {
            wb_repeat(b, L'\\', slashes * 2);
            break;
        }
        if (*p == L'"') {
            wb_repeat(b, L'\\', slashes * 2 + 1);
            wb_putc(b, L'"');
        } else {
            wb_repeat(b, L'\\', slashes);
            wb_putc(b, *p);
        }
    }
    wb_putc(b, L'"');
}

/* 子进程命令行：程序路径整体加引号（程序名不做转义解析），其余参数逐个转义。 */
static wchar_t *command_line(const wchar_t *program, int argc, wchar_t **argv)
{
    WBuf b = {0};
    int i;
    wb_putc(&b, L'"');
    wb_puts(&b, program);
    wb_putc(&b, L'"');
    for (i = 0; i < argc; ++i) {
        wb_putc(&b, L' ');
        append_quoted(&b, argv[i]);
    }
    return wb_take(&b);
}

/* ------------------------------------------------------------------ 两种启动方式 */

/* 内嵌。加载失败返回 0（可以换下一个候选或退回子进程）；成功调用了 Py_Main 返回 1。 */
static int run_embedded(const Candidate *c, const wchar_t *exe, const wchar_t *prog_dir, int pargc,
                        wchar_t **pargv, int *exit_code)
{
    wchar_t *dll = path_join(c->home, PYTHON_DLL);
    DLL_DIRECTORY_COOKIE cookie;
    HMODULE module;
    PyMainFn py_main = NULL;

    /* python3.dll、vcruntime140_1.dll 等由扩展模块间接加载，要能在安装目录里找到 */
    SetDllDirectoryW(c->home);
    cookie = AddDllDirectory(c->home);
    module = LoadLibraryExW(dll, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    free(dll);
    if (module)
        py_main = (PyMainFn)GetProcAddress(module, "Py_Main");
    if (!py_main) {
        if (module)
            FreeLibrary(module);
        if (cookie)
            RemoveDllDirectory(cookie);
        SetDllDirectoryW(NULL);
        return 0;
    }

    env_set(L"PYTHONHOME", c->home);
    env_set(L"PYTHONPATH", prog_dir);
    env_set(L"SPLENDER_EXE", exe);
    env_set(L"SPLENDER_LAUNCHER", L"embedded");
    *exit_code = py_main(pargc, pargv);
    return 1;
}

static int valid_handle(HANDLE h)
{
    return h != NULL && h != INVALID_HANDLE_VALUE;
}

/* 退回：pythonw.exe 子进程。创建失败返回 0 并给出错误码；否则等它结束，返回 1。 */
static int run_process(const Candidate *c, const wchar_t *exe, const wchar_t *prog_dir, int pargc, wchar_t **pargv,
                       int *exit_code, DWORD *error)
{
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    HANDLE std_in = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE std_out = GetStdHandle(STD_OUTPUT_HANDLE);
    HANDLE std_err = GetStdHandle(STD_ERROR_HANDLE);
    BOOL inherit = FALSE;
    DWORD code = 1;
    wchar_t *cmd;

    if (c->home && has_runtime(c->home))
        env_set(L"PYTHONHOME", c->home);
    env_set(L"PYTHONPATH", prog_dir);
    env_set(L"SPLENDER_EXE", exe);
    env_set(L"SPLENDER_LAUNCHER", L"process");

    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);
    /* 被别的程序带着管道调用时（比如安装检查读 --version 的输出），把标准句柄转交给子进程 */
    if (valid_handle(std_in) || valid_handle(std_out) || valid_handle(std_err)) {
        HANDLE handles[3];
        int i;
        handles[0] = std_in;
        handles[1] = std_out;
        handles[2] = std_err;
        for (i = 0; i < 3; ++i) {
            if (valid_handle(handles[i]))
                SetHandleInformation(handles[i], HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
        }
        si.dwFlags |= STARTF_USESTDHANDLES;
        si.hStdInput = valid_handle(std_in) ? std_in : NULL;
        si.hStdOutput = valid_handle(std_out) ? std_out : NULL;
        si.hStdError = valid_handle(std_err) ? std_err : NULL;
        inherit = TRUE;
    }

    cmd = command_line(c->pythonw, pargc - 1, pargv + 1);
    /* CREATE_NO_WINDOW：万一配置的是控制台版解释器，也不弹黑窗 */
    if (!CreateProcessW(c->pythonw, cmd, NULL, NULL, inherit, CREATE_NO_WINDOW, NULL, prog_dir, &si, &pi)) {
        *error = GetLastError();
        free(cmd);
        return 0;
    }
    free(cmd);
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    if (!GetExitCodeProcess(pi.hProcess, &code))
        code = 1;
    CloseHandle(pi.hProcess);
    *exit_code = (int)code;
    return 1;
}

/* ------------------------------------------------------------------ 出错提示 */

static void write_stderr(const wchar_t *text)
{
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD mode;
    DWORD written;
    int n;
    char *buf;
    if (!valid_handle(h))
        return;
    if (GetConsoleMode(h, &mode)) {
        WriteConsoleW(h, text, (DWORD)wcslen(text), &written, NULL);
        WriteConsoleW(h, L"\r\n", 2, &written, NULL);
        return;
    }
    n = WideCharToMultiByte(CP_UTF8, 0, text, -1, NULL, 0, NULL, NULL);
    if (n <= 1)
        return;
    buf = (char *)xmalloc((size_t)n);
    WideCharToMultiByte(CP_UTF8, 0, text, -1, buf, n, NULL, NULL);
    buf[n - 1] = '\n';
    WriteFile(h, buf, (DWORD)n, &written, NULL);
    free(buf);
}

static void enable_dpi_awareness(void)
{
    typedef BOOL(WINAPI * SetContextFn)(HANDLE);
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    SetContextFn set_context = NULL;
    if (user32)
        set_context = (SetContextFn)GetProcAddress(user32, "SetProcessDpiAwarenessContext");
    /* (HANDLE)-4 即 DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 */
    if (set_context && set_context((HANDLE)(LONG_PTR)-4))
        return;
    SetProcessDPIAware();
}

static void report(const wchar_t *text)
{
    write_stderr(text);
    if (env_flag(L"SPLENDER_NO_DIALOG"))
        return;
    enable_dpi_awareness();
    MessageBoxW(NULL, text, APP_TITLE, MB_OK | MB_ICONERROR | MB_SETFOREGROUND);
}

static wchar_t *system_message(DWORD error)
{
    wchar_t *raw = NULL;
    wchar_t *copy;
    DWORD n = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                 FORMAT_MESSAGE_IGNORE_INSERTS,
                             NULL, error, 0, (LPWSTR)&raw, 0, NULL);
    if (n == 0 || !raw)
        return NULL;
    while (n > 0 && (raw[n - 1] == L'\r' || raw[n - 1] == L'\n' || raw[n - 1] == L' '))
        raw[--n] = 0;
    copy = wdup(raw);
    LocalFree(raw);
    return copy;
}

typedef enum { MODE_AUTO, MODE_EMBED, MODE_PROCESS } LaunchMode;

static LaunchMode launch_mode(void)
{
    wchar_t *v = env_get(L"SPLENDER_LAUNCH_MODE");
    LaunchMode mode = MODE_AUTO;
    if (v) {
        if (_wcsicmp(v, L"embed") == 0)
            mode = MODE_EMBED;
        else if (_wcsicmp(v, L"process") == 0)
            mode = MODE_PROCESS;
        free(v);
    }
    return mode;
}

static void report_failure(const wchar_t *prog_dir, const wchar_t *override_home, LaunchMode mode,
                           const wchar_t *failed_program, DWORD error)
{
    WBuf b = {0};
    wchar_t *text;
    if (failed_program) {
        wchar_t *msg = system_message(error);
        wb_puts(&b, L"无法启动 Python 运行环境：\n");
        wb_puts(&b, failed_program);
        if (msg) {
            wb_puts(&b, L"\n\n");
            wb_puts(&b, msg);
            free(msg);
        }
    } else {
        wchar_t *runtime = path_join(prog_dir, BUNDLED_RUNTIME);
        wb_puts(&b, L"没有找到 Python 3.11 运行环境。\n\n");
        if (override_home) {
            wb_puts(&b, L"环境变量 SPLENDER_PYTHON_HOME 指向的目录里没有可用的 Python 3.11：\n");
            wb_puts(&b, override_home);
            wb_puts(&b, L"\n\n");
        }
        if (mode == MODE_EMBED)
            wb_puts(&b, L"环境变量 SPLENDER_LAUNCH_MODE 限定了只用内嵌方式启动。\n\n");
        else if (mode == MODE_PROCESS)
            wb_puts(&b, L"环境变量 SPLENDER_LAUNCH_MODE 限定了只用 pythonw.exe 启动。\n\n");
        wb_puts(&b, L"SPLENDER 需要 64 位的 Python 3.11。请到 https://www.python.org/downloads/ "
                    L"下载 Python 3.11 的「Windows installer (64-bit)」，安装时勾选"
                    L"「Add python.exe to PATH」，装好后再打开 SPLENDER。\n\n");
        wb_puts(&b, L"也可以把整套运行环境放到程序目录下：\n");
        wb_puts(&b, runtime);
        free(runtime);
    }
    text = wb_take(&b);
    report(text);
    free(text);
}

/* ------------------------------------------------------------------ 主流程 */

/* 让任务栏按 SPLENDER 自己分组、显示自己的图标。shell32 留在进程里，之后 Qt 也要用。 */
static void set_app_user_model_id(void)
{
    typedef HRESULT(WINAPI * SetIdFn)(PCWSTR);
    HMODULE shell32 = LoadLibraryExW(L"shell32.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
    SetIdFn set_id;
    if (!shell32)
        return;
    set_id = (SetIdFn)GetProcAddress(shell32, "SetCurrentProcessExplicitAppUserModelID");
    if (set_id)
        set_id(APP_USER_MODEL);
}

static int launcher_main(int argc, wchar_t **argv)
{
    wchar_t *exe;
    wchar_t *prog_dir;
    wchar_t *override_home;
    wchar_t **pargv;
    int pargc = 0;
    int user_argc = argc > 1 ? argc - 1 : 0;
    wchar_t **user_argv = argc > 1 ? argv + 1 : NULL;
    LaunchMode mode;
    Candidates list;
    const wchar_t *failed_program = NULL;
    DWORD error = 0;
    int i;

    set_app_user_model_id();

    exe = module_path();
    if (!exe) {
        report(L"无法确定 SPLENDER.exe 所在的位置。");
        return EXIT_START_FAILED;
    }
    prog_dir = parent_dir(exe);
    mode = launch_mode();
    override_home = env_get(L"SPLENDER_PYTHON_HOME");
    ZeroMemory(&list, sizeof(list));
    collect_candidates(&list, prog_dir, override_home);
    pargv = python_argv(exe, user_argc, user_argv, &pargc);

    if (mode != MODE_PROCESS) {
        for (i = 0; i < list.count; ++i) {
            int code;
            if (can_embed(&list.item[i]) && run_embedded(&list.item[i], exe, prog_dir, pargc, pargv, &code))
                return code;
        }
    }
    if (mode != MODE_EMBED) {
        for (i = 0; i < list.count; ++i) {
            const Candidate *c = &list.item[i];
            int code;
            if (!c->pythonw)
                continue;
            if (run_process(c, exe, prog_dir, pargc, pargv, &code, &error))
                return code;
            failed_program = c->pythonw;
        }
    }
    report_failure(prog_dir, override_home, mode, failed_program, error);
    return failed_program ? EXIT_START_FAILED : EXIT_NO_PYTHON;
}

#ifndef SPLENDER_LAUNCHER_NO_ENTRY
int WINAPI wWinMain(_In_ HINSTANCE instance, _In_opt_ HINSTANCE prev, _In_ LPWSTR cmd_line, _In_ int show)
{
    int argc = __argc;
    wchar_t **argv = __wargv;
    (void)instance;
    (void)prev;
    (void)cmd_line;
    (void)show;
    if (!argv) {
        /* C 运行库没给参数时自己解析，规则相同 */
        argv = CommandLineToArgvW(GetCommandLineW(), &argc);
        if (!argv)
            argc = 0;
    }
    return launcher_main(argc, argv);
}
#endif
