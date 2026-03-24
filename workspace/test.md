if response.get("action") == "create_file":
        path = response.get("path")
        if path:
            full_path = BASE_PATH / path
            if full_path.exists():
                response.update({
                    "action": "answer",
                    "message": "File already exists"
                })
            else:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.touch(exist_ok=True)
                if response.get("content"):
                    with open(full_path, "w") as f:
                        f.write(response.get("content"))
        else:
            response.update({"action": "answer", "message": "Path is required to create a file"})
    elif response.get("action") == "delete_file":
        path = response.get("path")
        if path:
            full_path = BASE_PATH / path
            if full_path.exists():
                if full_path.is_dir():
                    shutil.rmtree(full_path)
                else:
                    full_path.unlink()
            else:
                response.update({"message": f"Item at {path} doesn't exist"})
        else:
            response.update({"message": "Path is required for deletion"})
    elif response.get("action") == "rename_file":
        old_path = response.get("old_path")
        new_path = response.get("new_path")
        if old_path and new_path:
            full_old_path = BASE_PATH / old_path
            full_new_path = BASE_PATH / new_path
            if full_old_path.exists():
                full_old_path.rename(full_new_path)
            else:
                response.update({"message": f"Old path {old_path} doesn't exist"})
        else:
            response.update({"message": "Both old_path and new_path are required for renaming"})
    elif response.get("action") == "list_files":
        def get_all_items(path):
            items = []
            if not path.exists():
                return items
            for entry in path.iterdir():
                items.append(str(entry.relative_to(BASE_PATH)))
                if entry.is_dir():
                    items.extend(get_all_items(entry))
            return items
        all_items = get_all_items(BASE_PATH)
        response.update({
            "message": "Workspace items: " + (", ".join(all_items) if all_items else "Empty"),
            "items": all_items
        })
    elif response.get("action") == "answer":
        pass