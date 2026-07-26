from plantuml import PlantUML
import sys
import os

def main():
    server = PlantUML(url='http://www.plantuml.com/plantuml/img/')
    puml_file = 'docs/memory_process.puml'
    png_file = 'docs/memory_process.png'
    
    if not os.path.exists(puml_file):
        print(f"Error: {puml_file} not found")
        sys.exit(1)
        
    print(f"Generating {png_file} from {puml_file}...")
    server.processes_file(puml_file, outfile=png_file)
    print("Done.")

if __name__ == "__main__":
    main()
