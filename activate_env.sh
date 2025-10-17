#!/bin/bash

# Objetivo: Detectar y activar automáticamente un entorno virtual si hay uno solo,
# o mostrar una lista para seleccionar si hay varios, sin importar los nombres de las carpetas.
# Si no se encuentra ninguno, ofrecer la opción de crear un entorno .venv.

virtualenvs=()
virtualenv_names=()

# Buscar todos los entornos virtuales en subcarpetas (incluyendo carpetas que comienzan con punto)
for d in .*/ */ ; do
    # Saltar las carpetas especiales . y ..
    if [[ "$d" == "./" || "$d" == "../" ]]; then
        continue
    fi
    
    if [ -f "${d}bin/activate" ]; then
        virtualenvs+=("${d}")
        virtualenv_names+=("${d%/}")  # nombre sin la barra final
    fi
done

# Revisar si hay un entorno virtual en la carpeta actual (bin/activate)
if [ -f "bin/activate" ]; then
    virtualenvs+=("")
    virtualenv_names+=("(carpeta actual)")
fi

if [ ${#virtualenvs[@]} -eq 0 ]; then
    echo "No se encontró ningún entorno virtual en la carpeta actual ni en subcarpetas."
    read -p "¿Deseas crear un nuevo entorno virtual llamado '.venv' en la carpeta actual? (s/n): " create_env_choice
    if [[ "$create_env_choice" =~ ^[Ss]$ ]]; then
        echo "Creando entorno virtual '.venv'..."
        # Verificar si python3 está disponible
        if ! command -v python3 &> /dev/null; then
            echo "Error: 'python3' no está instalado o no se encuentra en el PATH. No se puede crear el entorno virtual."
            return 1 2>/dev/null || exit 1
        fi
        python3 -m venv .venv
        if [ $? -eq 0 ]; then
            echo "Entorno '.venv' creado exitosamente."
            echo "Activando entorno virtual: ./.venv"
            source ./.venv/bin/activate
            # El script terminará aquí si la activación es exitosa.
        else
            echo "Error al crear el entorno virtual '.venv'."
            return 1 2>/dev/null || exit 1
        fi
    else
        echo "No se creó ningún entorno virtual. Saliendo."
        return 1 2>/dev/null || exit 1
    fi
elif [ ${#virtualenvs[@]} -eq 1 ]; then
    if [ -z "${virtualenvs[0]}" ]; then
        echo "Activando entorno virtual en la carpeta actual."
        source "bin/activate"
    else
        echo "Activando entorno virtual: ${virtualenvs[0]}"
        source "${virtualenvs[0]}bin/activate"
    fi
else
    echo "Se encontraron múltiples entornos virtuales:"
    for i in "${!virtualenvs[@]}"; do
        echo "$((i+1))) ${virtualenv_names[$i]}"
    done
    while true; do
        echo -n "Selecciona el número del entorno virtual a activar: "
        read opt
        if [[ "$opt" =~ ^[0-9]+$ ]] && [ "$opt" -ge 1 ] && [ "$opt" -le "${#virtualenvs[@]}" ]; then
            env="${virtualenvs[$((opt-1))]}"
            if [ -z "$env" ]; then
                echo "Activando entorno virtual en la carpeta actual."
                source "bin/activate"
            else
                echo "Activando entorno virtual: $env"
                source "$env/bin/activate"
            fi
            break
        else
            echo "Opción inválida. Intenta de nuevo."
        fi
    done
fi