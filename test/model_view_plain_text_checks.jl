# Substring checks alone also pass inside Julia's default Node struct dump.
# Require the title and eager tab labels on actual lines, and reject renderer
# implementation details while checking content in each caller's fixture.
function test_model_view_plain_text(view, title, tabs)
    text = repr("text/plain", view)
    lines = split(text, '\n')
    @test first(lines) == title
    for label in tabs
        @test label in lines
    end
    @test !occursin(r"\bNode\(|_TabsetPart|OrderedDict|\bSemantic\w+\(", text)
    text
end
