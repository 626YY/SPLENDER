"""ModernGL 没有封装的 OpenGL 调用：稀疏贴图、同步对象、常驻映射缓冲、按层读回等。

必须在有当前上下文时使用。函数指针按需加载并缓存，共享上下文之间通用。
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, c_char_p, c_float, c_int, c_int64, c_size_t, c_ssize_t, c_ubyte, c_uint, c_uint64, c_void_p

# ---- 常量 ----
GL_TEXTURE_2D = 0x0DE1
GL_TEXTURE_2D_ARRAY = 0x8C1A
GL_RGBA8 = 0x8058
GL_SRGB8_ALPHA8 = 0x8C43
GL_R8 = 0x8229
GL_R16F = 0x822D
GL_RG16F = 0x822F
GL_RGBA16F = 0x881A
GL_RGBA32F = 0x8814
GL_RGBA = 0x1908
GL_RG = 0x8227
GL_RED = 0x1903
GL_UNSIGNED_BYTE = 0x1401
GL_HALF_FLOAT = 0x140B
GL_FLOAT = 0x1406

GL_TEXTURE_SPARSE_ARB = 0x91A6
GL_VIRTUAL_PAGE_SIZE_INDEX_ARB = 0x91A7
GL_NUM_VIRTUAL_PAGE_SIZES_ARB = 0x91A8
GL_VIRTUAL_PAGE_SIZE_X_ARB = 0x9195
GL_VIRTUAL_PAGE_SIZE_Y_ARB = 0x9196
GL_MAX_SPARSE_TEXTURE_SIZE_ARB = 0x9198

GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_TEXTURE_MAX_LEVEL = 0x813D
GL_TEXTURE_BASE_LEVEL = 0x813C
GL_TEXTURE_MAX_ANISOTROPY = 0x84FE
GL_NEAREST = 0x2600
GL_LINEAR = 0x2601
GL_LINEAR_MIPMAP_LINEAR = 0x2703
GL_CLAMP_TO_EDGE = 0x812F
GL_REPEAT = 0x2901

GL_FRAMEBUFFER = 0x8D40
GL_READ_FRAMEBUFFER = 0x8CA8
GL_DRAW_FRAMEBUFFER = 0x8CA9
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_FRAMEBUFFER_COMPLETE = 0x8CD5
GL_FRAMEBUFFER_SRGB = 0x8DB9
GL_FRAMEBUFFER_BINDING = 0x8CA6

GL_PIXEL_PACK_BUFFER = 0x88EB
GL_PIXEL_UNPACK_BUFFER = 0x88EC
GL_MAP_READ_BIT = 0x0001
GL_MAP_WRITE_BIT = 0x0002
GL_MAP_PERSISTENT_BIT = 0x0040
GL_MAP_COHERENT_BIT = 0x0080
GL_CLIENT_STORAGE_BIT = 0x0200
GL_PACK_ALIGNMENT = 0x0D05
GL_UNPACK_ALIGNMENT = 0x0CF5

GL_SYNC_GPU_COMMANDS_COMPLETE = 0x9117
GL_ALREADY_SIGNALED = 0x911A
GL_TIMEOUT_EXPIRED = 0x911B
GL_CONDITION_SATISFIED = 0x911C
GL_WAIT_FAILED = 0x911D
GL_SYNC_FLUSH_COMMANDS_BIT = 0x0001

GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX = 0x9047
GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX = 0x9048
GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX = 0x9049
GL_TEXTURE_FREE_MEMORY_ATI = 0x87FC
GL_MAX_ARRAY_TEXTURE_LAYERS = 0x88FF
GL_MAX_TEXTURE_SIZE = 0x0D33
GL_MAX_TEXTURE_IMAGE_UNITS = 0x8872
GL_NUM_EXTENSIONS = 0x821D
GL_EXTENSIONS = 0x1F03

_SIGNATURES = {
    "glGenTextures": (None, c_int, POINTER(c_uint)),
    "glDeleteTextures": (None, c_int, POINTER(c_uint)),
    "glBindTexture": (None, c_uint, c_uint),
    "glActiveTexture": (None, c_uint),
    "glTexParameteri": (None, c_uint, c_uint, c_int),
    "glTexParameterf": (None, c_uint, c_uint, c_float),
    "glTexStorage2D": (None, c_uint, c_int, c_uint, c_int, c_int),
    "glTexPageCommitmentARB": (None, c_uint, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_ubyte),
    "glGetInternalformativ": (None, c_uint, c_uint, c_uint, c_int, POINTER(c_int)),
    "glGetIntegerv": (None, c_uint, POINTER(c_int)),
    "glGetStringi": (c_char_p, c_uint, c_uint),
    "glGetError": (c_uint,),
    "glFlush": (None,),
    "glFinish": (None,),
    "glEnable": (None, c_uint),
    "glDisable": (None, c_uint),
    "glViewport": (None, c_int, c_int, c_int, c_int),
    "glColorMask": (None, c_ubyte, c_ubyte, c_ubyte, c_ubyte),
    "glGenFramebuffers": (None, c_int, POINTER(c_uint)),
    "glDeleteFramebuffers": (None, c_int, POINTER(c_uint)),
    "glBindFramebuffer": (None, c_uint, c_uint),
    "glFramebufferTexture2D": (None, c_uint, c_uint, c_uint, c_uint, c_int),
    "glFramebufferTextureLayer": (None, c_uint, c_uint, c_uint, c_int, c_int),
    "glDrawBuffers": (None, c_int, POINTER(c_uint)),
    "glCheckFramebufferStatus": (c_uint, c_uint),
    "glFenceSync": (c_void_p, c_uint, c_uint),
    "glClientWaitSync": (c_uint, c_void_p, c_uint, c_uint64),
    "glDeleteSync": (None, c_void_p),
    "glGenBuffers": (None, c_int, POINTER(c_uint)),
    "glDeleteBuffers": (None, c_int, POINTER(c_uint)),
    "glBindBuffer": (None, c_uint, c_uint),
    "glBufferStorage": (None, c_uint, c_ssize_t, c_void_p, c_uint),
    "glMapBufferRange": (c_void_p, c_uint, c_ssize_t, c_ssize_t, c_uint),
    "glUnmapBuffer": (c_ubyte, c_uint),
    "glPixelStorei": (None, c_uint, c_int),
    "glReadPixels": (None, c_int, c_int, c_int, c_int, c_uint, c_uint, c_void_p),
    "glGetTextureSubImage": (None, c_uint, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_uint, c_uint, c_int, c_void_p),
    "glTexSubImage3D": (None, c_uint, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_uint, c_uint, c_void_p),
    "glCopyImageSubData": (None, c_uint, c_uint, c_int, c_int, c_int, c_int, c_uint, c_uint, c_int, c_int, c_int, c_int,
                           c_int, c_int, c_int),
    "glMemoryBarrier": (None, c_uint),
    "glBindBufferBase": (None, c_uint, c_uint, c_uint),
    "glClearTexImage": (None, c_uint, c_int, c_uint, c_uint, c_void_p),
}

_functions: dict[str, object] = {}
_opengl32 = None


def _load(name: str):
    global _opengl32
    if _opengl32 is None:
        _opengl32 = ctypes.WinDLL("opengl32")
        _opengl32.wglGetProcAddress.restype = c_void_p
        _opengl32.wglGetProcAddress.argtypes = [c_char_p]
    signature = _SIGNATURES[name]
    address = _opengl32.wglGetProcAddress(name.encode("ascii"))
    # 驱动对 1.1 版函数返回空或小整数，这些从 opengl32.dll 本体取
    if not address or address in (1, 2, 3, 0xFFFFFFFFFFFFFFFF):
        try:
            func = getattr(_opengl32, name)
        except AttributeError:
            return None
        func.restype = signature[0]
        func.argtypes = list(signature[1:])
        return func
    return ctypes.WINFUNCTYPE(signature[0], *signature[1:])(address)


def fn(name: str):
    """取一个 GL 函数。需要有当前上下文。取不到返回 None。"""
    func = _functions.get(name)
    if func is None:
        func = _load(name)
        if func is not None:
            _functions[name] = func
    return func


def available(name: str) -> bool:
    return fn(name) is not None


# ---- 常用封装 ----
def get_int(pname: int) -> int:
    value = c_int(0)
    fn("glGetIntegerv")(pname, ctypes.byref(value))
    return int(value.value)


def error() -> int:
    return int(fn("glGetError")())


def clear_errors() -> None:
    get = fn("glGetError")
    for _ in range(16):
        if not get():
            break


def flush() -> None:
    fn("glFlush")()


def extensions() -> set[str]:
    count = get_int(GL_NUM_EXTENSIONS)
    get = fn("glGetStringi")
    found = set()
    for index in range(count):
        text = get(GL_EXTENSIONS, index)
        if text:
            found.add(text.decode("ascii", "replace"))
    return found


def vram_info_mb() -> tuple[int, int]:
    """(显存总量, 当前可用)，单位 MB。取不到返回 (0, 0)。"""
    clear_errors()
    total = get_int(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX)
    free = get_int(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX)
    if error() == 0 and total > 0:
        return total // 1024, free // 1024
    values = (c_int * 4)()
    fn("glGetIntegerv")(GL_TEXTURE_FREE_MEMORY_ATI, values)
    if error() == 0 and values[0] > 0:
        return 0, int(values[0]) // 1024
    return 0, 0


class Caps:
    """当前显卡与驱动的能力。"""

    def __init__(self) -> None:
        ext = extensions()
        self.extensions = ext
        self.sparse = "GL_ARB_sparse_texture" in ext and available("glTexPageCommitmentARB")
        self.sparse_clamp = "GL_ARB_sparse_texture_clamp" in ext
        self.buffer_storage = "GL_ARB_buffer_storage" in ext and available("glBufferStorage")
        self.get_sub_image = available("glGetTextureSubImage") and "GL_ARB_get_texture_sub_image" in ext
        self.draw_parameters = "GL_ARB_shader_draw_parameters" in ext
        self.anisotropy = "GL_EXT_texture_filter_anisotropic" in ext or "GL_ARB_texture_filter_anisotropic" in ext
        self.copy_image = "GL_ARB_copy_image" in ext
        self.max_array_layers = get_int(GL_MAX_ARRAY_TEXTURE_LAYERS)
        self.max_texture_size = get_int(GL_MAX_TEXTURE_SIZE)
        self.max_texture_units = get_int(GL_MAX_TEXTURE_IMAGE_UNITS)
        self.max_sparse_size = get_int(GL_MAX_SPARSE_TEXTURE_SIZE_ARB) if self.sparse else 0
        clear_errors()
        self.vram_total_mb, self.vram_free_mb = vram_info_mb()

    def sparse_page_size(self, internal_format: int) -> tuple[int, int]:
        count = c_int(0)
        fn("glGetInternalformativ")(GL_TEXTURE_2D, internal_format, GL_NUM_VIRTUAL_PAGE_SIZES_ARB, 1, ctypes.byref(count))
        if count.value <= 0:
            return (0, 0)
        xs = (c_int * count.value)()
        ys = (c_int * count.value)()
        fn("glGetInternalformativ")(GL_TEXTURE_2D, internal_format, GL_VIRTUAL_PAGE_SIZE_X_ARB, count.value, xs)
        fn("glGetInternalformativ")(GL_TEXTURE_2D, internal_format, GL_VIRTUAL_PAGE_SIZE_Y_ARB, count.value, ys)
        return (int(xs[0]), int(ys[0]))


# ---- 同步对象 ----
class Fence:
    """显卡命令完成的标记。ready() 不阻塞。"""

    __slots__ = ("handle",)

    def __init__(self) -> None:
        self.handle = fn("glFenceSync")(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

    def ready(self, wait_ns: int = 0) -> bool:
        if not self.handle:
            return True
        status = fn("glClientWaitSync")(self.handle, GL_SYNC_FLUSH_COMMANDS_BIT, wait_ns)
        return status in (GL_ALREADY_SIGNALED, GL_CONDITION_SATISFIED)

    def release(self) -> None:
        if self.handle:
            fn("glDeleteSync")(self.handle)
            self.handle = None


# ---- 稀疏贴图 ----
def create_sparse_texture(internal_format: int, size: int, levels: int, *, anisotropy: float = 1.0) -> int:
    """创建一张带多级渐远的稀疏二维贴图，返回 GL 对象号。此时不占显存。"""
    tex = c_uint(0)
    fn("glGenTextures")(1, ctypes.byref(tex))
    fn("glBindTexture")(GL_TEXTURE_2D, tex.value)
    fn("glTexParameteri")(GL_TEXTURE_2D, GL_TEXTURE_SPARSE_ARB, 1)
    fn("glTexParameteri")(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    fn("glTexParameteri")(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    fn("glTexParameteri")(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    fn("glTexParameteri")(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    fn("glTexParameteri")(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, levels - 1)
    if anisotropy > 1.0:
        fn("glTexParameterf")(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, float(anisotropy))
    fn("glTexStorage2D")(GL_TEXTURE_2D, levels, internal_format, size, size)
    fn("glBindTexture")(GL_TEXTURE_2D, 0)
    return int(tex.value)


def create_dense_texture(internal_format: int, size: int, levels: int, *, anisotropy: float = 1.0) -> int:
    tex = c_uint(0)
    fn("glGenTextures")(1, ctypes.byref(tex))
    fn("glBindTexture")(GL_TEXTURE_2D, tex.value)
    fn("glTexParameteri")(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    fn("glTexParameteri")(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    fn("glTexParameteri")(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    fn("glTexParameteri")(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    fn("glTexParameteri")(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, levels - 1)
    if anisotropy > 1.0:
        fn("glTexParameterf")(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, float(anisotropy))
    fn("glTexStorage2D")(GL_TEXTURE_2D, levels, internal_format, size, size)
    fn("glBindTexture")(GL_TEXTURE_2D, 0)
    return int(tex.value)


def set_anisotropy(texture: int, value: float) -> None:
    fn("glBindTexture")(GL_TEXTURE_2D, texture)
    fn("glTexParameterf")(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, float(max(1.0, value)))
    fn("glBindTexture")(GL_TEXTURE_2D, 0)


def commit_pages(texture: int, level: int, x: int, y: int, width: int, height: int, commit: bool) -> None:
    fn("glBindTexture")(GL_TEXTURE_2D, texture)
    fn("glTexPageCommitmentARB")(GL_TEXTURE_2D, level, x, y, 0, width, height, 1, 1 if commit else 0)


def delete_texture(texture: int) -> None:
    value = c_uint(texture)
    fn("glDeleteTextures")(1, ctypes.byref(value))


# ---- 帧缓冲（挂到贴图的指定层级）----
def create_framebuffer(textures: list[int], level: int) -> int:
    fbo = c_uint(0)
    fn("glGenFramebuffers")(1, ctypes.byref(fbo))
    fn("glBindFramebuffer")(GL_FRAMEBUFFER, fbo.value)
    buffers = (c_uint * len(textures))()
    for index, tex in enumerate(textures):
        fn("glFramebufferTexture2D")(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + index, GL_TEXTURE_2D, tex, level)
        buffers[index] = GL_COLOR_ATTACHMENT0 + index
    fn("glDrawBuffers")(len(textures), buffers)
    status = fn("glCheckFramebufferStatus")(GL_FRAMEBUFFER)
    fn("glBindFramebuffer")(GL_FRAMEBUFFER, 0)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("帧缓冲不完整：0x%X" % status)
    return int(fbo.value)


def bind_framebuffer(fbo: int, width: int, height: int, count: int = 1) -> None:
    fn("glBindFramebuffer")(GL_FRAMEBUFFER, fbo)
    buffers = (c_uint * count)(*[GL_COLOR_ATTACHMENT0 + i for i in range(count)])
    fn("glDrawBuffers")(count, buffers)
    fn("glViewport")(0, 0, width, height)
    fn("glColorMask")(1, 1, 1, 1)          # 颜色写入开关是全局状态，这里总是打开


def delete_framebuffer(fbo: int) -> None:
    value = c_uint(fbo)
    fn("glDeleteFramebuffers")(1, ctypes.byref(value))


def enable(cap: int, on: bool = True) -> None:
    (fn("glEnable") if on else fn("glDisable"))(cap)


# ---- 常驻映射的读回缓冲 ----
class PersistentBuffer:
    """显卡把像素读进这块缓冲，工作线程直接从映射地址取数据，主线程不做拷贝。"""

    def __init__(self, size: int) -> None:
        self.size = size
        handle = c_uint(0)
        fn("glGenBuffers")(1, ctypes.byref(handle))
        self.handle = int(handle.value)
        fn("glBindBuffer")(GL_PIXEL_PACK_BUFFER, self.handle)
        flags = GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_CLIENT_STORAGE_BIT
        fn("glBufferStorage")(GL_PIXEL_PACK_BUFFER, size, None, flags)
        self.address = fn("glMapBufferRange")(GL_PIXEL_PACK_BUFFER, 0, size, GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT)
        fn("glBindBuffer")(GL_PIXEL_PACK_BUFFER, 0)
        if not self.address:
            raise RuntimeError("读回缓冲映射失败")
        self.memory = (ctypes.c_ubyte * size).from_address(self.address)

    def view(self, offset: int, length: int) -> memoryview:
        return memoryview(self.memory)[offset: offset + length]

    def release(self) -> None:
        if self.handle:
            fn("glBindBuffer")(GL_PIXEL_PACK_BUFFER, self.handle)
            fn("glUnmapBuffer")(GL_PIXEL_PACK_BUFFER)
            fn("glBindBuffer")(GL_PIXEL_PACK_BUFFER, 0)
            value = c_uint(self.handle)
            fn("glDeleteBuffers")(1, ctypes.byref(value))
            self.handle = 0


def read_layer_into(buffer: PersistentBuffer, offset: int, texture: int, layer: int, width: int, height: int,
                    gl_format: int, gl_type: int, nbytes: int) -> None:
    """把贴图数组的一层读进常驻缓冲的 offset 处（异步，之后用 Fence 判断完成）。"""
    fn("glBindBuffer")(GL_PIXEL_PACK_BUFFER, buffer.handle)
    fn("glPixelStorei")(GL_PACK_ALIGNMENT, 1)
    fn("glGetTextureSubImage")(texture, 0, 0, 0, layer, width, height, 1, gl_format, gl_type, nbytes, c_void_p(offset))
    fn("glBindBuffer")(GL_PIXEL_PACK_BUFFER, 0)


GL_SHADER_STORAGE_BUFFER = 0x90D2
GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT = 0x00004000


def clear_texture(texture: int, gl_format: int, gl_type: int) -> None:
    """把整张贴图清零（新建的页池数组先清一遍，第一次使用的代价就不会落在画画的那一帧）。"""
    if available("glClearTexImage"):
        fn("glClearTexImage")(texture, 0, gl_format, gl_type, None)


def bind_storage(binding: int, buffer_handle: int) -> None:
    """把一块常驻映射的缓冲当作着色器存储缓冲绑定。"""
    fn("glBindBufferBase")(GL_SHADER_STORAGE_BUFFER, binding, buffer_handle)


def client_barrier() -> None:
    """着色器写进常驻映射缓冲的数据，要在之后的 Fence 完成时对 CPU 可见。"""
    fn("glMemoryBarrier")(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT)


def copy_layer(src_texture: int, src_layer: int, dst_texture: int, dst_layer: int, width: int, height: int) -> None:
    """在显卡上把贴图数组的一层拷到另一层（两边格式相同）。"""
    fn("glCopyImageSubData")(src_texture, GL_TEXTURE_2D_ARRAY, 0, 0, 0, src_layer,
                             dst_texture, GL_TEXTURE_2D_ARRAY, 0, 0, 0, dst_layer, width, height, 1)
