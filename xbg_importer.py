bl_info = {
    "name": "XBG Importer",
    "author": "Quiet Joker",
    "version": (1, 0, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > XBG Import",
    "description": "Import XBG models from James Cameron's Avatar The Game",
    "category": "Import-Export",
}

import bpy
import struct
import os
import math
import mathutils
from typing import List, Tuple, Optional, Dict, Any
import re

# =============================================================================
# BINARY READER
# =============================================================================

class BinaryReader:
    """Binary reader for game files"""
    
    def __init__(self, file_path: str):
        self.file = open(file_path, 'rb')
        self.endian = '<'
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
    
    def tell(self) -> int:
        return self.file.tell()
    
    def seek(self, offset: int, whence: int = 0):
        self.file.seek(offset, whence)
    
    def seekpad(self, pad: int, type: int = 0):
        """16-byte chunk alignment"""
        size = self.file.tell()
        seek = (pad - (size % pad)) % pad
        if type == 1:
            if seek == 0:
                seek += pad
        self.file.seek(seek, 1)
    
    def i(self, n: int) -> Tuple:
        return struct.unpack(self.endian + n * 'i', self.file.read(n * 4))
    
    def I(self, n: int) -> Tuple:
        return struct.unpack(self.endian + n * 'I', self.file.read(n * 4))
    
    def h(self, n: int) -> Tuple:
        return struct.unpack(self.endian + n * 'h', self.file.read(n * 2))
    
    def H(self, n: int) -> Tuple:
        return struct.unpack(self.endian + n * 'H', self.file.read(n * 2))
    
    def f(self, n: int) -> Tuple:
        return struct.unpack(self.endian + n * 'f', self.file.read(n * 4))
    
    def B(self, n: int) -> Tuple:
        return struct.unpack(self.endian + n * 'B', self.file.read(n))
    
    def b(self, n: int) -> Tuple:
        return struct.unpack(self.endian + n * 'b', self.file.read(n))
    
    def word(self, length: int) -> str:
        s = ''
        for j in range(length):
            lit = struct.unpack('c', self.file.read(1))[0]
            if ord(lit) != 0:
                s += lit.decode('utf-8', errors='ignore')
        return s

# =============================================================================
# MATH STRUCTURES (Internal representation for parsing)
# =============================================================================

class Vector:
    def __init__(self, x=0, y=0, z=0):
        if isinstance(x, (list, tuple)) and len(x) >= 3:
            self.x, self.y, self.z = x[0], x[1], x[2]
        else:
            self.x, self.y, self.z = x, y, z
    
    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

class CustomMatrix4x4:
    """Row-Major storage for parsing logic"""
    def __init__(self, data: List[float] = None):
        if data is None:
            self.matrix = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        else:
            self.matrix = [list(data[i:i+4]) for i in range(0, 16, 4)]
    
    def multiply(self, other: 'CustomMatrix4x4') -> 'CustomMatrix4x4':
        result = CustomMatrix4x4()
        m1 = self.matrix
        m2 = other.matrix
        for i in range(4):
            for j in range(4):
                result.matrix[i][j] = sum(m1[i][k] * m2[k][j] for k in range(4))
        return result
    
    def get_translation(self) -> List[float]:
        return [self.matrix[0][3], self.matrix[1][3], self.matrix[2][3]]

class CustomQuaternion:
    def __init__(self, x: float, y: float, z: float, w: float):
        self.x, self.y, self.z, self.w = x, y, z, w
    
    def to_matrix4x4(self) -> CustomMatrix4x4:
        x, y, z, w = self.x, self.y, self.z, self.w
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        mat = CustomMatrix4x4()
        mat.matrix[0][0] = 1 - 2 * (yy + zz)
        mat.matrix[0][1] = 2 * (xy - wz)
        mat.matrix[0][2] = 2 * (xz + wy)
        mat.matrix[1][0] = 2 * (xy + wz)
        mat.matrix[1][1] = 1 - 2 * (xx + zz)
        mat.matrix[1][2] = 2 * (yz - wx)
        mat.matrix[2][0] = 2 * (xz - wy)
        mat.matrix[2][1] = 2 * (yz + wx)
        mat.matrix[2][2] = 1 - 2 * (xx + yy)
        return mat

def quaternion_from_xbg_data(quat_data: List[float]) -> CustomQuaternion:
    if len(quat_data) >= 4:
        return CustomQuaternion(quat_data[0], quat_data[1], quat_data[2], quat_data[3])
    return CustomQuaternion(0, 0, 0, 1)

def create_translation_matrix(position: List[float]) -> CustomMatrix4x4:
    mat = CustomMatrix4x4()
    mat.matrix[0][3] = position[0]
    mat.matrix[1][3] = position[1]
    mat.matrix[2][3] = position[2]
    return mat

# =============================================================================
# MESH & SKELETON STRUCTURES
# =============================================================================

class MeshPrimitive:
    def __init__(self):
        self.indices: List[int] = []
        self.material_index: int = 0
        self.material_name: str = "Default"

class Mesh:
    def __init__(self):
        self.vert_pos_list: List[List[float]] = []
        self.vert_uv_list: List[List[float]] = []
        self.primitives: List[MeshPrimitive] = []
        self.mat_list_info: List[Tuple] = []
        self.skin_weight_list: List[Tuple] = []
        self.skin_indice_list: List[Tuple] = []
        self.vert_count: int = 0
        self.face_count: int = 0
        self.vert_stride: int = 0
        self.vert_section_offset: int = 0
        self.indice_section_offset: int = 0
        self.lod_level: int = 0 

    def add_primitive(self, indices: List[int], mat_idx: int, mat_name: str):
        prim = MeshPrimitive()
        prim.indices = indices
        prim.material_index = mat_idx
        prim.material_name = mat_name
        self.primitives.append(prim)

class SubMesh:
    def __init__(self):
        self.header_data: List[int] = []
        self.bone_data: List[int] = []
        self.face_count: int = 0
        
    def get_face_count(self) -> int:
        if len(self.header_data) > 1:
            return self.header_data[1]
        return 0

class Bone:
    def __init__(self):
        self.name: Optional[str] = None
        self.parent_id: Optional[int] = None
        
        # Raw data
        self.local_rotation_quat: Optional[CustomQuaternion] = None
        self.local_position: List[float] = [0, 0, 0]
        
        # Matrices
        self.local_matrix: Optional[CustomMatrix4x4] = None
        self.world_matrix: Optional[CustomMatrix4x4] = None

class Skeleton:
    def __init__(self):
        self.bones: List[Bone] = []
        
    def add_bone(self, bone: Bone):
        self.bones.append(bone)
        
    def get_bone_count(self) -> int:
        return len(self.bones)
    
    def compute_bone_transforms(self):
        for i, bone in enumerate(self.bones):
            if bone.local_rotation_quat is None:
                continue
            
            rot_matrix = bone.local_rotation_quat.to_matrix4x4()
            pos_matrix = create_translation_matrix(bone.local_position)
            
            # Local = Translation * Rotation
            bone.local_matrix = pos_matrix.multiply(rot_matrix)
            
            if bone.parent_id is not None and 0 <= bone.parent_id < len(self.bones):
                parent = self.bones[bone.parent_id]
                if parent.world_matrix is not None:
                    bone.world_matrix = parent.world_matrix.multiply(bone.local_matrix)
                else:
                    bone.world_matrix = bone.local_matrix
            else:
                bone.world_matrix = bone.local_matrix

# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_mesh_vertices(g, mesh: Mesh, vert_pos_scale: float, uv_trans: float, uv_scale: float):
    g.seek(mesh.vert_section_offset)
    
    for m in range(mesh.vert_count):
        tm = g.tell()
        
        # Read vertex position
        pos_data = g.h(3)
        pos = Vector(pos_data) * vert_pos_scale
        mesh.vert_pos_list.append(pos.to_list())
        
        g.h(1)  # skip
        
        # Read UV coordinates
        u = uv_trans + g.h(1)[0] * uv_scale
        v = (uv_trans + g.h(1)[0] * uv_scale)
        # Blender UVs are typically Y-up, check if 1.0 - v is needed later
        mesh.vert_uv_list.append([u, 1.0 - v]) 
        
        g.seek(4, 1)  # skip 4 bytes
        
        # Read skinning data
        if mesh.vert_stride == 40:
            mesh.skin_weight_list.append(g.B(4))
            mesh.skin_indice_list.append(g.B(4))
        
        g.seek(tm + mesh.vert_stride)

def parse_skeleton_chunk(g, skeleton: Skeleton):
    w = g.i(3)
    bone_count = w[2]
    
    for m in range(bone_count):
        bone = Bone()
        g.b(4)
        w = g.i(3)
        
        quat_data = g.f(4)
        bone.local_rotation_quat = quaternion_from_xbg_data(quat_data)
        
        pos_data = g.f(3)
        bone.local_position = list(pos_data)
        
        g.f(3); g.i(1); g.f(1); g.i(1) # skip padding
        
        name_len = g.i(1)[0]
        bone.name = g.word(name_len)[-25:] 
        bone.parent_id = w[2]
        g.b(1)
        
        skeleton.add_bone(bone)
    
    skeleton.compute_bone_transforms()

class XBGData:
    def __init__(self):
        self.skeleton = Skeleton()
        self.meshes: List[Mesh] = []
        self.sub_mesh_list: List[List[SubMesh]] = []
        self.materials: List[str] = []
        self.lod_count: int = 0
        self.vert_pos_scale: float = 1.0
        self.uv_trans: float = 0.0
        self.uv_scale: float = 1.0

class XBGParser:
    def __init__(self, filename: str):
        self.filename = filename
        self.data = XBGData()
        
    def parse(self, lod_level: int = 0) -> XBGData:
        with BinaryReader(self.filename) as g:
            g.word(4) # Header
            header_data = g.i(7)
            chunk_count = header_data[6]
            
            for m in range(chunk_count):
                back = g.tell()
                chunk = g.word(4)
                chunk_info = g.i(2)
                
                if chunk == 'PMCP':
                    g.i(2)
                    unk, self.data.vert_pos_scale = g.f(2)
                elif chunk == 'PMCU':
                    g.i(2)
                    self.data.uv_trans, self.data.uv_scale = g.f(2)
                elif chunk == 'EDON':
                    parse_skeleton_chunk(g, self.data.skeleton)
                elif chunk == 'DIKS':
                    g.i(2)
                    self.data.lod_count = g.i(1)[0]
                    for m in range(self.data.lod_count): g.H(2); g.B(4)
                elif chunk == 'LTMR':
                    w = g.i(4)
                    mat_count = w[2]
                    for m in range(mat_count):
                        name_len = g.i(1)[0]
                        mat_file = g.word(name_len)
                        simple_name = mat_file.split('/')[-1].replace('.mat', '') or f"Material_{m}"
                        self.data.materials.append(simple_name)
                        g.b(1)
                elif chunk == 'SDOL':
                    self._parse_sdol(g)
                elif chunk == 'DNKS':
                    self._parse_dnks(g)
                
                g.seek(back + chunk_info[1])
            
            self._filter_lod(lod_level)
            self._process_mesh_vertices(g)
            self._remap_skin_indices(g)
            self._process_mesh_faces(g)
            
        return self.data

    def _parse_sdol(self, g):
        g.i(2)
        lod_count = g.i(1)[0]
        for m in range(lod_count):
            mesh = Mesh()
            mesh.lod_level = m
            w = g.i(6)
            mesh.face_count = w[1]; mesh.vert_count = w[4]; mesh.vert_stride = w[3]
            count = g.i(1)[0]
            mesh.mat_list_info = [g.i(7) for _ in range(count)]
            
            vert_section_size = g.I(1)[0]
            g.seekpad(16)
            mesh.vert_section_offset = g.tell()
            g.seek(mesh.vert_section_offset + vert_section_size)
            
            indice_section_size = g.I(1)[0]
            g.seekpad(16)
            mesh.indice_section_offset = g.tell()
            g.seek(mesh.indice_section_offset + indice_section_size * 2)
            self.data.meshes.append(mesh)

    def _parse_dnks(self, g):
        g.i(2); g.word(4); g.i(4)
        self.data.sub_mesh_list = []
        if self.data.lod_count == 0: return
        
        for n in range(self.data.lod_count):
            lod_submeshes = []
            mat_count = g.i(1)[0]
            for m in range(mat_count):
                submesh = SubMesh()
                submesh.header_data = list(g.H(7))
                submesh.bone_data = list(g.h(48))
                submesh.face_count = submesh.get_face_count()
                lod_submeshes.append(submesh)
            self.data.sub_mesh_list.append(lod_submeshes)

    def _filter_lod(self, lod_level):
        # -1 indicates importing ALL LODs
        if lod_level == -1:
            return 
            
        if lod_level >= 0 and lod_level < len(self.data.meshes):
            self.data.meshes = [self.data.meshes[lod_level]]
        elif self.data.meshes:
            self.data.meshes = [self.data.meshes[0]]

    def _process_mesh_vertices(self, g):
        for mesh in self.data.meshes:
            parse_mesh_vertices(g, mesh, self.data.vert_pos_scale, self.data.uv_trans, self.data.uv_scale)

    def _remap_skin_indices(self, g):
        for mesh in self.data.meshes:
            if not mesh.skin_indice_list or not mesh.mat_list_info: continue
            vert_id_start = 0
            for info in mesh.mat_list_info:
                lod_grp, sub_idx = info[1], info[2]
                if lod_grp < len(self.data.sub_mesh_list):
                    submesh = self.data.sub_mesh_list[lod_grp][sub_idx] if sub_idx < len(self.data.sub_mesh_list[lod_grp]) else None
                    if submesh:
                        count = submesh.header_data[5]
                        palette = submesh.bone_data
                        end = min(vert_id_start + count, len(mesh.skin_indice_list))
                        for v_idx in range(vert_id_start, end):
                            mesh.skin_indice_list[v_idx] = tuple((palette[r] if r < len(palette) and palette[r] != -1 else 0) for r in mesh.skin_indice_list[v_idx])
                        vert_id_start += count

    def _process_mesh_faces(self, g):
        for mesh in self.data.meshes:
            for info in mesh.mat_list_info:
                lod_grp, sub_idx = info[1], info[2]
                if lod_grp < len(self.data.sub_mesh_list) and sub_idx < len(self.data.sub_mesh_list[lod_grp]):
                    submesh = self.data.sub_mesh_list[lod_grp][sub_idx]
                    mat_id = submesh.header_data[0]
                    mat_name = self.data.materials[mat_id] if mat_id < len(self.data.materials) else f"Material_{mat_id}"
                    
                    if submesh.face_count > 0:
                        g.seek(mesh.indice_section_offset + info[3] * 2)
                        indices = []
                        for _ in range(submesh.face_count):
                            try:
                                fi = g.H(3)
                                if 65535 not in fi: indices.extend(fi)
                            except: break
                        if indices: mesh.add_primitive(indices, mat_id, mat_name)

# =============================================================================
# XBM MATERIAL & XBT TEXTURE PARSING
# =============================================================================

class XBMMaterialData:
    """Parsed data from an XBM material file"""
    def __init__(self):
        self.textures: Dict[str, str] = {}  # type -> path (e.g., 'diffuse' -> 'graphics/...')
        self.illumination_color: Optional[Tuple[float, float, float]] = None  # Normalized RGB
        self.diffuse_tiling: float = 1.0
        self.specular_tiling: float = 1.0
        self.normal_tiling: float = 1.0

class XBMParser:
    """Parser for XBM material files"""
    
    @staticmethod
    def parse(filepath: str, load_hd_textures: bool = True) -> Optional[XBMMaterialData]:
        """Parse an XBM file and extract texture paths and material properties"""
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            result = XBMMaterialData()
            
            # Extract texture paths
            XBMParser._extract_textures(data, result)
            
            # Extract IlluminationColor1
            XBMParser._extract_illumination_color(data, result)
            
            # Extract tiling values
            XBMParser._extract_tiling(data, result)
            
            # Look for missing textures in the same directory as found textures
            XBMParser._find_missing_textures(result, filepath, load_hd_textures)
            
            return result
        except Exception as e:
            print(f"Error parsing XBM file {filepath}: {e}")
            return None
    
    @staticmethod
    def _extract_textures(data: bytes, result: XBMMaterialData):
        """Extract texture paths from XBM data"""
        i = 0
        # Dictionary to track both mip0 and non-mip0 versions
        found_textures = {}  # tex_type -> {'mip0': path or None, 'regular': path or None}
        
        while i < len(data) - 20:
            # Look for graphics\ or graphics/ path patterns
            if data[i:i+9] == b'graphics\\' or data[i:i+9] == b'graphics/':
                path_start = i
                path_end = data.find(b'\x00', path_start)
                
                if path_end != -1 and path_end - path_start < 200:
                    path = data[path_start:path_end].decode('ascii', errors='ignore')
                    
                    if path.endswith('.xbt') or path.lower().endswith('.xbt'):
                        # Determine texture type from filename suffix
                        basename = os.path.basename(path).lower()
                        
                        # Determine if this is a mip0 variant
                        is_mip0 = '_mip0.xbt' in basename
                        
                        # Check for texture type suffixes
                        tex_type = None
                        if '_d.xbt' in basename or '_d_mip0.xbt' in basename:
                            tex_type = 'diffuse'
                        elif '_n.xbt' in basename or '_n_mip0.xbt' in basename:
                            tex_type = 'normal'
                        elif '_s.xbt' in basename or '_s_mip0.xbt' in basename:
                            tex_type = 'specular'
                        elif '_m.xbt' in basename or '_m_mip0.xbt' in basename:
                            tex_type = 'bio'
                        
                        if tex_type:
                            # Initialize texture type tracking if needed
                            if tex_type not in found_textures:
                                found_textures[tex_type] = {'mip0': None, 'regular': None}
                            
                            # Store the path based on whether it's mip0 or regular
                            if is_mip0:
                                found_textures[tex_type]['mip0'] = path
                                print(f"  Found {tex_type} (mip0): {path}")
                            else:
                                found_textures[tex_type]['regular'] = path
                                print(f"  Found {tex_type}: {path}")
                        else:
                            # Unknown type, store with path as key
                            print(f"  Found unknown texture: {path}")
                            result.textures[basename] = path
                    
                    i = path_end + 1
                else:
                    i += 1
            else:
                i += 1
        
        # Now decide which version to use for each texture type (prefer mip0)
        for tex_type, versions in found_textures.items():
            if versions['mip0']:
                result.textures[tex_type] = versions['mip0']
                print(f"  → Using mip0 version for {tex_type}")
            elif versions['regular']:
                result.textures[tex_type] = versions['regular']
                print(f"  → Using regular version for {tex_type}")

    
    @staticmethod
    def _extract_illumination_color(data: bytes, result: XBMMaterialData):
        """Extract IlluminationColor1 RGB values"""
        # Look for IlluminationColor1 string
        search_terms = [b'IlluminationColor1', b'illuminationcolor1']
        
        for term in search_terms:
            pos = data.find(term)
            if pos != -1:
                # Value should be after the null terminator
                val_pos = pos + len(term)
                # Skip null terminators
                while val_pos < len(data) and data[val_pos] == 0:
                    val_pos += 1
                
                # Read 3 floats (RGB)
                if val_pos + 12 <= len(data):
                    try:
                        r = struct.unpack('<f', data[val_pos:val_pos+4])[0]
                        g = struct.unpack('<f', data[val_pos+4:val_pos+8])[0]
                        b = struct.unpack('<f', data[val_pos+8:val_pos+12])[0]
                        
                        # Normalize HDR values to 0-1 range
                        max_val = max(r, g, b, 1.0)
                        if max_val > 0:
                            result.illumination_color = (r / max_val, g / max_val, b / max_val)
                        else:
                            result.illumination_color = (0.0, 0.0, 0.0)
                        return
                    except:
                        pass
    
    @staticmethod
    def _extract_tiling(data: bytes, result: XBMMaterialData):
        """Extract tiling values for diffuse, specular, and normal maps"""
        tiling_props = [
            (b'DiffuseTiling1', 'diffuse_tiling'),
            (b'SpecularTiling1', 'specular_tiling'),
            (b'NormalTiling1', 'normal_tiling'),
        ]
        
        for search_term, attr_name in tiling_props:
            pos = data.find(search_term)
            if pos != -1:
                val_pos = pos + len(search_term)
                # Skip null terminators
                while val_pos < len(data) and data[val_pos] == 0:
                    val_pos += 1
                
                # Read float value
                if val_pos + 4 <= len(data):
                    try:
                        value = struct.unpack('<f', data[val_pos:val_pos+4])[0]
                        # Only set if it's a reasonable value
                        if 0.001 < abs(value) < 1000:
                            setattr(result, attr_name, value)
                    except:
                        pass
    
    @staticmethod
    def _find_missing_textures(result: XBMMaterialData, xbm_filepath: str, load_hd_textures: bool = True):
        """Look for missing texture types AND mip0 variants in the same directory
        
        If the XBM has some textures but not others (e.g., _n and _m but no _d),
        check if those missing textures exist in the same directory with matching base names.
        Also checks for _mip0 variants of existing textures.
        """
        # Get the data folder from XBM path (up to 'graphics/_materials')
        xbm_dir = os.path.dirname(xbm_filepath)
        # Navigate up to find the Data folder
        data_folder = xbm_dir
        while data_folder and not data_folder.endswith('Data'):
            parent = os.path.dirname(data_folder)
            if parent == data_folder:  # Reached root
                break
            data_folder = parent
        
        if not data_folder or not os.path.exists(data_folder):
            print("  Could not determine data folder for texture search")
            return
        
        # Get any texture we found to extract base name and directory
        reference_texture = None
        for tex_type in ['diffuse', 'normal', 'specular', 'bio']:
            if tex_type in result.textures:
                reference_texture = result.textures[tex_type]
                break
        
        if not reference_texture:
            return  # No textures found at all
        
        # Extract base name (remove _d, _n, _s, _m suffix)
        basename = os.path.basename(reference_texture).lower()
        # Remove .xbt extension
        basename = basename.replace('.xbt', '')
        # Remove suffixes
        for suffix in ['_d', '_n', '_s', '_m', '_mip0']:
            if basename.endswith(suffix):
                basename = basename[:-len(suffix)]
                break
        
        # Get the directory where textures are
        texture_dir = os.path.dirname(reference_texture)
        
        print(f"  Searching for missing textures with base name: {basename}")
        print(f"  In directory: {texture_dir}")
        
        # Check for missing texture types AND mip0 variants
        texture_types = [
            ('diffuse', '_d.xbt', '_d_mip0.xbt'),
            ('normal', '_n.xbt', '_n_mip0.xbt'),
            ('specular', '_s.xbt', '_s_mip0.xbt'),
            ('bio', '_m.xbt', '_m_mip0.xbt')
        ]
        
        for tex_type, suffix, mip0_suffix in texture_types:
            # First check if we need to find a mip0 variant for existing texture
            if tex_type in result.textures:
                current_path = result.textures[tex_type]
                # Only check for mip0 if HD textures are enabled and current texture isn't already mip0
                if load_hd_textures and '_mip0.xbt' not in current_path.lower():
                    # Build potential mip0 path
                    potential_filename = basename + mip0_suffix
                    potential_path = texture_dir + '/' + potential_filename
                    
                    # Check if this file exists
                    full_path = os.path.join(data_folder, potential_path.replace('\\', os.sep).replace('/', os.sep))
                    
                    if os.path.exists(full_path):
                        result.textures[tex_type] = potential_path
                        print(f"  ✓ Upgraded to mip0 version for {tex_type}: {potential_filename}")
                    # Silently skip if no mip0 found (avoid spam)
            else:
                # Texture type is missing, try to find it
                # If HD textures enabled, prefer mip0, otherwise use regular
                if load_hd_textures:
                    mip0_filename = basename + mip0_suffix
                    mip0_path = texture_dir + '/' + mip0_filename
                    mip0_full_path = os.path.join(data_folder, mip0_path.replace('\\', os.sep).replace('/', os.sep))
                    
                    if os.path.exists(mip0_full_path):
                        result.textures[tex_type] = mip0_path
                        print(f"  ✓ Found missing {tex_type} texture (mip0): {mip0_filename}")
                        continue
                
                # Try regular texture (either HD disabled or mip0 not found)
                regular_filename = basename + suffix
                regular_path = texture_dir + '/' + regular_filename
                regular_full_path = os.path.join(data_folder, regular_path.replace('\\', os.sep).replace('/', os.sep))
                
                if os.path.exists(regular_full_path):
                    result.textures[tex_type] = regular_path
                    print(f"  ✓ Found missing {tex_type} texture: {regular_filename}")
                else:
                    print(f"  ✗ Missing {tex_type} texture not found")


class XBTConverter:
    """Convert XBT files to DDS format for loading into Blender"""
    
    # Cache for converted DDS files (path -> temp file path)
    _temp_files: Dict[str, str] = {}
    
    @staticmethod
    def get_temp_dds_path(xbt_path: str) -> Optional[str]:
        """Get a temporary DDS file path for an XBT file"""
        import tempfile
        
        # Check cache first
        if xbt_path in XBTConverter._temp_files:
            temp_path = XBTConverter._temp_files[xbt_path]
            if os.path.exists(temp_path):
                return temp_path
        
        # Convert to DDS
        dds_data = XBTConverter.convert_to_dds(xbt_path)
        if dds_data is None:
            return None
        
        # Write to temp file
        try:
            # Create a unique temp file name
            basename = os.path.splitext(os.path.basename(xbt_path))[0]
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"xbg_import_{basename}_{hash(xbt_path) & 0xFFFFFFFF}.dds")
            
            with open(temp_path, 'wb') as f:
                f.write(dds_data)
            
            # Cache the path
            XBTConverter._temp_files[xbt_path] = temp_path
            return temp_path
        except Exception as e:
            print(f"Error writing temp DDS file: {e}")
            return None
    
    @staticmethod
    def cleanup_temp_files():
        """Clean up temporary DDS files"""
        for path in XBTConverter._temp_files.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
        XBTConverter._temp_files.clear()
    
    @staticmethod
    def convert_to_dds(xbt_path: str) -> Optional[bytes]:
        """Strip XBT header and return raw DDS data"""
        try:
            with open(xbt_path, 'rb') as f:
                xbt_data = f.read()
            
            # Check for XBT/TBX header
            if xbt_data[:3] == b'TBX':
                # Read header size from offset 8
                if len(xbt_data) >= 12:
                    header_size = struct.unpack('<I', xbt_data[8:12])[0]
                    
                    # Verify header size is reasonable
                    if 32 <= header_size <= 1024 and header_size < len(xbt_data):
                        dds_data = xbt_data[header_size:]
                    else:
                        # Try default 32 byte header
                        dds_data = xbt_data[32:]
                else:
                    dds_data = xbt_data[32:]
            else:
                # No XBT header, might already be DDS
                dds_data = xbt_data
            
            # Verify DDS signature
            if len(dds_data) >= 4 and dds_data[:4] == b'DDS ':
                return dds_data
            
            # Try to find DDS header at various offsets
            for offset in [64, 128, 256]:
                if len(xbt_data) > offset:
                    test_data = xbt_data[offset:]
                    if len(test_data) >= 4 and test_data[:4] == b'DDS ':
                        return test_data
            
            return None
        except Exception as e:
            print(f"Error converting XBT to DDS: {e}")
            return None
    
    @staticmethod
    def find_mip0_variant(texture_path: str, data_folder: str) -> Optional[str]:
        """Find the _mip0 (high quality) variant of a texture if it exists"""
        if '_mip0.xbt' in texture_path.lower():
            return texture_path  # Already mip0
        
        mip0_path = texture_path.replace('.xbt', '_mip0.xbt')
        full_path = os.path.join(data_folder, mip0_path.replace('\\', os.sep).replace('/', os.sep))
        
        if os.path.exists(full_path):
            return mip0_path
        return None


class BlenderMaterialSetup:
    """Setup Blender materials with textures from XBM data"""
    
    @staticmethod
    def setup_material(mat: Any, xbm_data: XBMMaterialData, data_folder: str, load_hd_textures: bool = True):
        """Setup a Blender material with textures and nodes"""
        if not mat.use_nodes:
            mat.use_nodes = True
        
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Find or create principled BSDF
        bsdf = None
        for node in nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf = node
                break
        
        if not bsdf:
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            bsdf.location = (0, 0)
        
        # Find material output
        output = None
        for node in nodes:
            if node.type == 'OUTPUT_MATERIAL':
                output = node
                break
        
        if not output:
            output = nodes.new('ShaderNodeOutputMaterial')
            output.location = (300, 0)
        
        # Connect BSDF to output if not connected
        if not bsdf.outputs['BSDF'].links:
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        # Track texture positions for layout
        tex_y_offset = 300
        
        # Check if we need tiling nodes
        needs_tiling = (xbm_data.diffuse_tiling != 1.0 or 
                       xbm_data.specular_tiling != 1.0 or 
                       xbm_data.normal_tiling != 1.0)
        
        tex_coord = None
        mapping_nodes = {}
        
        if needs_tiling:
            # Create Texture Coordinate node
            tex_coord = nodes.new('ShaderNodeTexCoord')
            tex_coord.location = (-1200, 0)
            
            # Create mapping nodes for each tiling value
            for tex_type, tiling in [('diffuse', xbm_data.diffuse_tiling),
                                     ('specular', xbm_data.specular_tiling),
                                     ('normal', xbm_data.normal_tiling)]:
                if tiling != 1.0:
                    mapping = nodes.new('ShaderNodeMapping')
                    mapping.location = (-1000, tex_y_offset)
                    mapping.inputs['Scale'].default_value = (tiling, tiling, 1.0)
                    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
                    mapping_nodes[tex_type] = mapping
                    tex_y_offset -= 200
        
        # Setup each texture type
        tex_y_offset = 300
        
        # 1. Diffuse texture -> Base Color
        if 'diffuse' in xbm_data.textures:
            tex_y_offset = BlenderMaterialSetup._setup_diffuse(
                nodes, links, bsdf, xbm_data.textures['diffuse'], 
                data_folder, mapping_nodes.get('diffuse'), tex_y_offset, load_hd_textures
            )
        
        # 2. Specular texture -> IOR Level (non-color)
        if 'specular' in xbm_data.textures:
            tex_y_offset = BlenderMaterialSetup._setup_specular(
                nodes, links, bsdf, xbm_data.textures['specular'],
                data_folder, mapping_nodes.get('specular'), tex_y_offset, load_hd_textures
            )
        
        # 3. Normal map -> Special reconstruction setup
        if 'normal' in xbm_data.textures:
            tex_y_offset = BlenderMaterialSetup._setup_normal(
                nodes, links, bsdf, xbm_data.textures['normal'],
                data_folder, mapping_nodes.get('normal'), tex_y_offset, load_hd_textures
            )
        
        # 4. Bio/Emission mask -> Emission with IlluminationColor1
        if 'bio' in xbm_data.textures:
            # Check if we should use illumination color or connect directly
            use_color_multiply = False
            if xbm_data.illumination_color:
                r, g, b = xbm_data.illumination_color
                # Skip color multiply if it's pure black (0,0,0) or pure white (1,1,1)
                # In these cases, the _m texture already contains the color
                # Use tolerance for both checks due to floating-point precision
                is_black = (abs(r) < 0.01 and abs(g) < 0.01 and abs(b) < 0.01)
                is_white = (abs(r - 1.0) < 0.01 and abs(g - 1.0) < 0.01 and abs(b - 1.0) < 0.01)
                
                if is_black:
                    print(f"  Skipping emission multiply - color is pure black (0,0,0)")
                elif is_white:
                    print(f"  Skipping emission multiply - color is pure white (1,1,1)")
                else:
                    print(f"  Using emission color multiply: RGB({r:.3f}, {g:.3f}, {b:.3f})")
                
                use_color_multiply = not (is_black or is_white)
            
            tex_y_offset = BlenderMaterialSetup._setup_bio_emission(
                nodes, links, bsdf, xbm_data.textures['bio'],
                xbm_data.illumination_color if use_color_multiply else None,
                data_folder, tex_y_offset, load_hd_textures
            )
    
    @staticmethod
    def _load_texture_node(nodes, texture_path: str, data_folder: str, 
                          location: tuple, non_color: bool = False, load_hd_textures: bool = True) -> Optional[Any]:
        """Create and load a texture node"""
        # Only look for _mip0 variant if HD textures are enabled
        if load_hd_textures:
            mip0_path = XBTConverter.find_mip0_variant(texture_path, data_folder)
            actual_path = mip0_path if mip0_path else texture_path
        else:
            actual_path = texture_path
        
        # Get full path
        full_path = os.path.join(data_folder, actual_path.replace('\\', os.sep).replace('/', os.sep))
        
        if not os.path.exists(full_path):
            print(f"Texture not found: {full_path}")
            return None
        
        # Convert XBT to DDS
        dds_path = XBTConverter.get_temp_dds_path(full_path)
        if not dds_path:
            print(f"Failed to convert XBT: {full_path}")
            return None
        
        # Create unique image name to avoid duplicates
        img_name = os.path.basename(texture_path)
        
        # Check if image already exists in Blender
        img = bpy.data.images.get(img_name)
        
        if not img:
            # Load image for the first time
            try:
                img = bpy.data.images.load(dds_path)
                img.name = img_name
                
                # Pack the image into the blend file so it's not dependent on temp files
                img.pack()
                
                if non_color:
                    img.colorspace_settings.name = 'Non-Color'
            except Exception as e:
                print(f"Failed to load texture {dds_path}: {e}")
                return None
        
        # Create image texture node
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = location
        tex_node.image = img
        
        return tex_node
    
    @staticmethod
    def _setup_diffuse(nodes, links, bsdf, texture_path: str, data_folder: str,
                      mapping_node: Optional[Any], y_offset: int, load_hd_textures: bool = True) -> int:
        """Setup diffuse texture -> Base Color and Alpha"""
        tex_node = BlenderMaterialSetup._load_texture_node(
            nodes, texture_path, data_folder, (-600, y_offset), non_color=False, load_hd_textures=load_hd_textures
        )
        
        if tex_node:
            if mapping_node:
                links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
            
            # Connect color to base color
            links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
            
            # Always connect alpha to BSDF alpha
            links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
        
        return y_offset - 300
    
    @staticmethod
    def _setup_specular(nodes, links, bsdf, texture_path: str, data_folder: str,
                       mapping_node: Optional[Any], y_offset: int, load_hd_textures: bool = True) -> int:
        """Setup specular texture -> IOR Level (non-color)"""
        tex_node = BlenderMaterialSetup._load_texture_node(
            nodes, texture_path, data_folder, (-600, y_offset), non_color=True, load_hd_textures=load_hd_textures
        )
        
        if tex_node:
            if mapping_node:
                links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
            # Connect to IOR Level (Specular in older Blender versions)
            if 'IOR Level' in bsdf.inputs:
                links.new(tex_node.outputs['Color'], bsdf.inputs['IOR Level'])
            elif 'Specular IOR Level' in bsdf.inputs:
                links.new(tex_node.outputs['Color'], bsdf.inputs['Specular IOR Level'])
            elif 'Specular' in bsdf.inputs:
                links.new(tex_node.outputs['Color'], bsdf.inputs['Specular'])
        
        return y_offset - 300
    
    @staticmethod
    def _setup_normal(nodes, links, bsdf, texture_path: str, data_folder: str,
                     mapping_node: Optional[Any], y_offset: int, load_hd_textures: bool = True) -> int:
        """Setup normal map with Avatar's special reconstruction:
        - Texture Color output -> Combine Color Green channel
        - Texture Alpha output -> Combine Color Red channel  
        - Blue channel = 1.0
        - Combine output -> Normal Map node -> BSDF
        """
        tex_node = BlenderMaterialSetup._load_texture_node(
            nodes, texture_path, data_folder, (-900, y_offset), non_color=True, load_hd_textures=load_hd_textures
        )
        
        if tex_node:
            if mapping_node:
                links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
            
            # Create Combine Color node (or Combine RGB for older Blender)
            try:
                combine_node = nodes.new('ShaderNodeCombineColor')
            except:
                combine_node = nodes.new('ShaderNodeCombineRGB')
            combine_node.location = (-600, y_offset)
            
            # Connect texture Color output directly to Green channel
            links.new(tex_node.outputs['Color'], combine_node.inputs[1])  # Color -> Green
            
            # Connect texture Alpha output to Red channel
            links.new(tex_node.outputs['Alpha'], combine_node.inputs[0])  # Alpha -> Red
            
            # Set Blue channel to 1.0
            combine_node.inputs[2].default_value = 1.0  # Blue = 1.0
            
            # Create Normal Map node
            normal_map = nodes.new('ShaderNodeNormalMap')
            normal_map.location = (-300, y_offset)
            normal_map.inputs['Strength'].default_value = 1.0
            
            # Connect Combine output -> Normal Map Color input
            links.new(combine_node.outputs[0], normal_map.inputs['Color'])
            
            # Connect Normal Map -> BSDF Normal
            links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
        
        return y_offset - 400
    
    @staticmethod
    def _setup_bio_emission(nodes, links, bsdf, texture_path: str,
                           illumination_color: Optional[Tuple[float, float, float]],
                           data_folder: str, y_offset: int, load_hd_textures: bool = True) -> int:
        """Setup bio emission mask with optional IlluminationColor1:
        - If illumination_color is provided: Multiply texture by color, then to Emission
        - If illumination_color is None: Connect texture directly to Emission (texture has color baked in)
        """
        tex_node = BlenderMaterialSetup._load_texture_node(
            nodes, texture_path, data_folder, (-600, y_offset), non_color=False, load_hd_textures=load_hd_textures
        )
        
        if tex_node:
            if illumination_color:
                # Create Mix/Multiply node to combine texture with illumination color
                try:
                    # Blender 3.4+
                    multiply_node = nodes.new('ShaderNodeMix')
                    multiply_node.data_type = 'RGBA'
                    multiply_node.blend_type = 'MULTIPLY'
                    multiply_node.inputs['Factor'].default_value = 1.0
                except:
                    # Older Blender
                    multiply_node = nodes.new('ShaderNodeMixRGB')
                    multiply_node.blend_type = 'MULTIPLY'
                    multiply_node.inputs['Fac'].default_value = 1.0
                
                multiply_node.location = (-300, y_offset)
                
                # Connect bio texture to A input
                # Check socket naming for Blender 5.0 compatibility
                if 'A' in multiply_node.inputs:
                    links.new(tex_node.outputs['Color'], multiply_node.inputs['A'])
                elif 'Color1' in multiply_node.inputs:
                    links.new(tex_node.outputs['Color'], multiply_node.inputs['Color1'])
                else:
                    # Fallback to first color input
                    links.new(tex_node.outputs['Color'], multiply_node.inputs[6])
                
                # Set B input to IlluminationColor1
                if 'B' in multiply_node.inputs:
                    multiply_node.inputs['B'].default_value = (*illumination_color, 1.0)
                elif 'Color2' in multiply_node.inputs:
                    multiply_node.inputs['Color2'].default_value = (*illumination_color, 1.0)
                else:
                    # Fallback to second color input
                    multiply_node.inputs[7].default_value = (*illumination_color, 1.0)
                
                # Connect multiply output to Emission
                # Check socket naming for Blender 5.0 compatibility
                if 'Result' in multiply_node.outputs:
                    output_socket = multiply_node.outputs['Result']
                elif 'Color' in multiply_node.outputs:
                    output_socket = multiply_node.outputs['Color']
                else:
                    # Fallback to first output
                    output_socket = multiply_node.outputs[0]
                
                # Connect to emission
                if 'Emission Color' in bsdf.inputs:
                    links.new(output_socket, bsdf.inputs['Emission Color'])
                elif 'Emission' in bsdf.inputs:
                    links.new(output_socket, bsdf.inputs['Emission'])
                
                print(f"  Bio emission with color multiply: {illumination_color}")
            else:
                # Connect texture directly to emission (color is baked into texture)
                if 'Emission Color' in bsdf.inputs:
                    links.new(tex_node.outputs['Color'], bsdf.inputs['Emission Color'])
                elif 'Emission' in bsdf.inputs:
                    links.new(tex_node.outputs['Color'], bsdf.inputs['Emission'])
                
                print(f"  Bio emission direct (color baked into texture)")
            
            # Set emission strength
            if 'Emission Strength' in bsdf.inputs:
                bsdf.inputs['Emission Strength'].default_value = 1.0
        
        return y_offset - 300


# =============================================================================
# BLENDER IMPORT LOGIC
# =============================================================================

class XBGBlenderImporter:
    def load(self, context, filepath, lod_level=0, import_mesh_only=False,
             data_folder="", load_textures=True, load_hd_textures=True, flip_normals=True):
        print(f"Starting import: {filepath}")
        parser = XBGParser(filepath)
        data = parser.parse(lod_level)
        
        armature_obj = None
        
        # 1. Create Armature (only if not mesh-only mode)
        if not import_mesh_only:
            armature_obj = self.create_armature(data.skeleton, os.path.basename(filepath))
        
        # 2. Create Meshes
        mesh_objects = self.create_meshes(data.meshes, armature_obj, data.materials, 
                                          import_mesh_only, data_folder, load_textures, load_hd_textures)
        
        # 3. Flip normals if requested
        if flip_normals and mesh_objects:
            self.flip_mesh_normals(mesh_objects)
        
        # 4. Cleanup temp files
        XBTConverter.cleanup_temp_files()

        return {'FINISHED'}

    def create_armature(self, skeleton: Skeleton, name_base: str):
        if skeleton.get_bone_count() == 0:
            return None
        
        # Create Armature Data and Object
        amt_name = f"{name_base}_Armature"
        amt_data = bpy.data.armatures.new(amt_name)
        amt_obj = bpy.data.objects.new(amt_name, amt_data)
        bpy.context.collection.objects.link(amt_obj)
        bpy.context.view_layer.objects.active = amt_obj
        
        # 2. ROTATION FIX: Set to 180 degrees on Z to face forward, remove X rotation
        amt_obj.rotation_euler = (0, 0, math.radians(180))

        bpy.ops.object.mode_set(mode='EDIT')
        
        # Temporary dictionary to store edit bones
        edit_bones = {}
        
        # Pass 1: Create bones and set heads
        for i, bone_data in enumerate(skeleton.bones):
            b_name = bone_data.name if bone_data.name else f"Bone_{i}"
            eb = amt_data.edit_bones.new(b_name)
            edit_bones[i] = eb
            
            # Get translation from computed world matrix
            if bone_data.world_matrix:
                trans = bone_data.world_matrix.get_translation()
                eb.head = mathutils.Vector(trans)
            else:
                eb.head = mathutils.Vector((0, 0, 0))
                
            # Set a default tail (will be fixed by children or orientation)
            # 3. BONE SCALING FIX: Increased from 0.1 to 0.5 (5x)
            eb.tail = eb.head + mathutils.Vector((0, 0.5, 0))

        # Pass 2: Hierarchy and Orientation
        for i, bone_data in enumerate(skeleton.bones):
            eb = edit_bones[i]
            
            # Parent
            if bone_data.parent_id is not None and bone_data.parent_id in edit_bones:
                eb.parent = edit_bones[bone_data.parent_id]
                eb.use_connect = False
            
            # Orientation (Apply the rotation)
            if bone_data.world_matrix:
                # Extract rotation from our custom matrix
                m = bone_data.world_matrix.matrix
                bl_mat = mathutils.Matrix(m)
                
                rot = bl_mat.to_quaternion()
                
                # 3. BONE SCALING FIX: Apply the 5x scale here too for rotated bones
                length = 0.5 
                offset = mathutils.Vector((0, 1, 0)) * length
                offset.rotate(rot)
                eb.tail = eb.head + offset
        
        bpy.ops.object.mode_set(mode='OBJECT')
        return amt_obj

    def create_meshes(self, meshes: List[Mesh], armature_obj, material_names: List[str], 
                       import_mesh_only=False, data_folder="", load_textures=True, load_hd_textures=True) -> List[Any]:
        """Create Blender mesh objects and return the list of created objects"""
        created_objects = []
        
        for m_idx, mesh in enumerate(meshes):
            if not mesh.vert_pos_list:
                continue
                
            mesh_name = f"Mesh_LOD{mesh.lod_level}_{m_idx}"
            
            # Create Blender Mesh
            me = bpy.data.meshes.new(mesh_name)
            obj = bpy.data.objects.new(mesh_name, me)
            bpy.context.collection.objects.link(obj)
            created_objects.append(obj)
            
            # If Mesh Only mode, apply the forward rotation to the mesh directly
            if import_mesh_only:
                 obj.rotation_euler = (0, 0, math.radians(180))
            
            # Parent to Armature
            if armature_obj:
                obj.parent = armature_obj
                modifier = obj.modifiers.new(name="Armature", type='ARMATURE')
                modifier.object = armature_obj
            
            # Prepare Geometry
            verts = mesh.vert_pos_list
            faces = []
            
            # Materials
            bl_materials = []
            # Map material index to blender material index (0, 1, 2...)
            mat_mapping = {} 
            # Track materials for texture loading
            materials_to_setup = []
            
            for prim in mesh.primitives:
                # Get or Create Material
                mat_idx = prim.material_index
                if mat_idx not in mat_mapping:
                    mat_real_name = prim.material_name
                    if not mat_real_name and mat_idx < len(material_names):
                        mat_real_name = material_names[mat_idx]
                    
                    # Check if exists
                    mat = bpy.data.materials.get(mat_real_name)
                    if not mat:
                        mat = bpy.data.materials.new(name=mat_real_name)
                        mat.use_nodes = True
                    
                    obj.data.materials.append(mat)
                    mat_mapping[mat_idx] = len(obj.data.materials) - 1
                    
                    # Store material info for texture loading
                    materials_to_setup.append((mat, mat_real_name))
                
                bl_mat_index = mat_mapping[mat_idx]
                
                # Process Indices (Triangles)
                for i in range(0, len(prim.indices), 3):
                    if i + 2 < len(prim.indices):
                        poly = (prim.indices[i], prim.indices[i+1], prim.indices[i+2])
                        faces.append(poly)
            
            # Create Mesh Data
            me.from_pydata(verts, [], faces)
            me.update()
            
            # Assign Material Indices to Polygons
            poly_offset = 0
            for prim in mesh.primitives:
                bl_mat_index = mat_mapping.get(prim.material_index, 0)
                num_tris = len(prim.indices) // 3
                for i in range(num_tris):
                    if poly_offset + i < len(me.polygons):
                        me.polygons[poly_offset + i].material_index = bl_mat_index
                poly_offset += num_tris

            # UVs
            if mesh.vert_uv_list:
                uv_layer = me.uv_layers.new(name="UVMap")
                for loop in me.loops:
                    uv_layer.data[loop.index].uv = mesh.vert_uv_list[loop.vertex_index]
            
            # Weights / Vertex Groups (Only if armature exists)
            if armature_obj and mesh.skin_indice_list and mesh.skin_weight_list:
                bone_names = [b.name for b in armature_obj.data.bones]
                v_groups = {name: obj.vertex_groups.new(name=name) for name in bone_names}
                
                for v_idx, (indices, weights) in enumerate(zip(mesh.skin_indice_list, mesh.skin_weight_list)):
                    for b_idx_idx, b_id in enumerate(indices):
                        weight = weights[b_idx_idx] / 255.0
                        if weight > 0.0:
                            if b_id < len(bone_names):
                                b_name = bone_names[b_id]
                                if b_name in v_groups:
                                    v_groups[b_name].add([v_idx], weight, 'REPLACE')
            
            # Setup textures for materials
            if load_textures and data_folder:
                self.setup_material_textures(materials_to_setup, data_folder, load_hd_textures)
        
        return created_objects
    
    def setup_material_textures(self, materials_to_setup: List[Tuple], data_folder: str, load_hd_textures: bool = True):
        """Setup textures for materials by finding and parsing XBM files"""
        materials_folder = os.path.join(data_folder, "graphics", "_materials")
        
        for mat, mat_name in materials_to_setup:
            # The material name from XBG is like "GRAPHICS\_MATERIALS\CSEAUT-M-2009041439268299.xbm"
            # We need to extract just the filename and find it
            
            # Parse the material name to get the XBM path
            xbm_path = None
            
            # Check if it's a full path reference
            if "\\" in mat_name or "/" in mat_name:
                # Extract just the filename
                xbm_filename = os.path.basename(mat_name)
                # Ensure it has .xbm extension
                if not xbm_filename.lower().endswith('.xbm'):
                    xbm_filename += '.xbm'
                xbm_path = os.path.join(materials_folder, xbm_filename)
            else:
                # Try direct filename
                xbm_filename = mat_name
                if not xbm_filename.lower().endswith('.xbm'):
                    xbm_filename += '.xbm'
                xbm_path = os.path.join(materials_folder, xbm_filename)
            
            # Check if XBM file exists
            if xbm_path and os.path.exists(xbm_path):
                print(f"Found XBM material: {xbm_path}")
                
                # Parse the XBM file with HD texture setting
                xbm_data = XBMParser.parse(xbm_path, load_hd_textures)
                
                if xbm_data:
                    # Setup the material with textures
                    BlenderMaterialSetup.setup_material(mat, xbm_data, data_folder, load_hd_textures)
                    print(f"  Loaded textures: {list(xbm_data.textures.keys())}")
                else:
                    print(f"  Failed to parse XBM file")
            else:
                print(f"XBM file not found: {xbm_path}")
    
    def flip_mesh_normals(self, mesh_objects: List[Any]):
        """Flip all face normals on the given mesh objects"""
        for obj in mesh_objects:
            if obj.type != 'MESH':
                continue
            
            # Use bmesh to flip normals
            import bmesh
            
            # Get mesh data
            me = obj.data
            
            # Create bmesh from mesh
            bm = bmesh.new()
            bm.from_mesh(me)
            
            # Flip normals
            bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
            
            # Write back to mesh
            bm.to_mesh(me)
            bm.free()
            
            # Update mesh
            me.update()
            
            print(f"Flipped normals for: {obj.name}")

# =============================================================================
# ADDON PROPERTIES & PREFERENCES
# =============================================================================

class XBGAddonPreferences(bpy.types.AddonPreferences):
    """Addon preferences for persistent settings"""
    bl_idname = __name__
    
    data_folder: bpy.props.StringProperty(
        name="Data Folder",
        description="Path to the game's Data folder (e.g., D:\\Games\\Avatar The Game\\Data_Win32\\Data)",
        default="",
        subtype='DIR_PATH'
    )
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "data_folder")

class XBGImportSettings(bpy.types.PropertyGroup):
    """Settings for XBG import stored in scene"""
    load_textures: bpy.props.BoolProperty(
        name="Load Textures",
        description="Automatically load and setup textures from XBM material files",
        default=True
    )
    load_hd_textures: bpy.props.BoolProperty(
        name="Load HD Textures",
        description="Use high-resolution _mip0 texture variants when available (requires Load Textures)",
        default=True
    )
    flip_normals: bpy.props.BoolProperty(
        name="Flip Normals",
        description="Flip all face normals after import (fixes inverted normals)",
        default=True
    )

# =============================================================================
# UI PANELS & OPERATORS
# =============================================================================

class XBG_OT_Import(bpy.types.Operator):
    """Import XBG Model"""
    bl_idname = "import_scene.xbg_model"
    bl_label = "Import XBG"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    # 4. UI Option: Import Mesh Only
    import_mesh_only: bpy.props.BoolProperty(
        name="Import Mesh Only",
        description="Skip skeleton import and rigging",
        default=False
    )
    
    # 5. UI Option: Import All LODs
    import_all_lods: bpy.props.BoolProperty(
        name="Import All LODs",
        description="Import all Level of Details found in file",
        default=False
    )
    
    lod_level: bpy.props.IntProperty(
        name="LOD Level", 
        default=0, 
        min=0,
        description="Specific LOD to import if 'All LODs' is unchecked"
    )
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        if not self.filepath.endswith(".xbg") and not self.filepath.endswith(".XBG"):
            self.report({'ERROR'}, "File must be an .xbg file")
            return {'CANCELLED'}
        
        # Get settings from scene and preferences
        settings = context.scene.xbg_settings
        prefs = context.preferences.addons[__name__].preferences
        data_folder = prefs.data_folder
        load_textures = settings.load_textures
        load_hd_textures = settings.load_hd_textures
        flip_normals = settings.flip_normals
        
        # Validate data folder if texture loading is enabled
        if load_textures and not data_folder:
            self.report({'WARNING'}, "Data folder not set - textures will not be loaded")
            load_textures = False
        elif load_textures and not os.path.exists(data_folder):
            self.report({'WARNING'}, f"Data folder does not exist: {data_folder}")
            load_textures = False
            
        importer = XBGBlenderImporter()
        
        # Logic to handle LOD selection
        target_lod = -1 if self.import_all_lods else self.lod_level
        
        importer.load(context, self.filepath, target_lod, self.import_mesh_only,
                     data_folder, load_textures, load_hd_textures, flip_normals)
        
        # Report success
        if load_textures:
            self.report({'INFO'}, f"Imported XBG with textures")
        else:
            self.report({'INFO'}, f"Imported XBG (no textures)")
        
        return {'FINISHED'}

class XBG_PT_Panel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "XBG Import"
    bl_idname = "OBJECT_PT_xbg_import"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.xbg_settings
        prefs = context.preferences.addons[__name__].preferences
        
        # Data folder path (from preferences)
        box = layout.box()
        box.label(text="Game Data Folder:", icon='FILE_FOLDER')
        box.prop(prefs, "data_folder", text="")
        
        # Import options (from scene settings)
        box = layout.box()
        box.label(text="Import Options:", icon='PREFERENCES')
        box.prop(settings, "load_textures")
        # Indent the HD textures option and disable it if Load Textures is off
        row = box.row()
        row.enabled = settings.load_textures
        row.prop(settings, "load_hd_textures")
        box.prop(settings, "flip_normals")
        
        # Import button
        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("import_scene.xbg_model", icon='IMPORT')

# =============================================================================
# REGISTRATION
# =============================================================================

classes = (
    XBGAddonPreferences,
    XBGImportSettings,
    XBG_OT_Import,
    XBG_PT_Panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.xbg_settings = bpy.props.PointerProperty(type=XBGImportSettings)

def unregister():
    del bpy.types.Scene.xbg_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()