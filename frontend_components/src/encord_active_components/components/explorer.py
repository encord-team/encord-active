from typing import List, TypedDict

from encord_active_components.renderer import Components, render



class Image(TypedDict):
    url: str

# class OutputAction(str, Enum):
#     INIT = "INIT"


def explorer(images: List[Image] = []):
    return render(component=Components.EXPLORER, props={"images": images})


if __name__ == "__main__":
    pass
    # key = pages_menu(ITEMS)
