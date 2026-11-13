"""A provider plugin used by the tests of the `python` provider type."""


class EchoModel:
    def __init__(self, options):
        self.prefix = options.get("prefix", "")

    def chat(self, system, user, schema=None, max_output=1500):
        return f'{{"answer": "{self.prefix}{len(user)}"}}'     # a plain string is allowed


def make(options):
    return EchoModel(options)


class WithCapabilities(EchoModel):
    capabilities = {"remote": False, "uses_local_gpu": True}


class BrokenDescribe(EchoModel):
    def describe_model(self):
        raise RuntimeError("model gone. Run `pull it`.")


NOT_CALLABLE = 5
