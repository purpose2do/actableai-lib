from .deci import DeciRunner
from .direct_lingam import DirectLiNGAMRunner
from .notears import NotearsRunner
from .pc import PCRunner

runners = {
    "deci": DeciRunner,
    "notears": NotearsRunner,
    "direct-lingam": DirectLiNGAMRunner,
    "pc": PCRunner,
}
