import os

def listar_cursos(base_path="documentos"):
    """
    Lista las carpetas disponibles dentro de 'documentos' como cursos.
    """
    if not os.path.exists(base_path):
        return []
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def seleccionar_curso():
    """
    Muestra un menú para que el usuario seleccione la carpeta de curso.
    """
    cursos = listar_cursos()
    if not cursos:
        print("❌ No se encontraron carpetas de cursos en la carpeta 'documentos'.")
        return None

    print("\n📚 Cursos disponibles:")
    for i, curso in enumerate(cursos):
        print(f"  [{i + 1}] {curso}")

    while True:
        try:
            opcion = int(input("\nSelecciona el curso (número): "))
            if 1 <= opcion <= len(cursos):
                return f"documentos/{cursos[opcion - 1]}"
        except ValueError:
            pass
        print("❌ Opción inválida. Intenta de nuevo.")
